#!/usr/bin/env python3
"""
Full GRID Dataset Training Script
===================================
Optimized for A100 80GB + 128-core CPU + 503GB RAM

Usage:
    python train_full.py --data-dir ./processed_data_full --epochs 30 --batch-size 128
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler

import cv2
from tqdm import tqdm

try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

# ============================================================================
# CONFIGURATION - Optimized for A100 80GB
# ============================================================================

CHARSET = "abcdefghijklmnopqrstuvwxyz '"
BLANK_IDX = 0
char2idx = {ch: idx + 1 for idx, ch in enumerate(CHARSET)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
NUM_CLASSES = len(CHARSET) + 1

CONFIG = {
    "batch_size": 128,           # Large batch for A100
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "epochs": 30,
    "max_frames": 75,
    "frame_height": 50,
    "frame_width": 100,
    "cnn_output_dim": 256,
    "lstm_hidden_size": 512,
    "lstm_layers": 2,
    "dropout": 0.3,
    "gradient_clip": 1.0,
    "warmup_epochs": 3,
    "val_split": 0.1,
    "test_split": 0.1,
    "num_workers": 16,           # More workers for 128-core CPU
    "pin_memory": True,
    "use_amp": True,
}

# ============================================================================
# DATASET
# ============================================================================

class LipReadingDataset(Dataset):
    def __init__(self, manifest_path, max_frames=75, frame_size=(50, 100)):
        self.max_frames = max_frames
        self.frame_height, self.frame_width = frame_size
        
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        print(f"Dataset: {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        lips_dir = Path(sample["lips_dir"])
        frame_files = sorted(lips_dir.glob("*.jpg"))
        
        frames = []
        for frame_path in frame_files[:self.max_frames]:
            img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (self.frame_width, self.frame_height))
                img = img.astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5
                frames.append(img)
        
        num_frames = len(frames)
        while len(frames) < self.max_frames:
            frames.append(frames[-1] if frames else np.zeros((self.frame_height, self.frame_width), dtype=np.float32))
        
        video = np.stack(frames[:self.max_frames])
        video = np.expand_dims(video, axis=1)
        video = torch.from_numpy(video).float()
        
        transcript = sample["transcript"]
        label = [char2idx[c] for c in transcript if c in char2idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return video, label, num_frames, len(label)


def collate_fn(batch):
    videos, labels, input_lengths, label_lengths = zip(*batch)
    videos = torch.stack(videos)
    flat_labels = torch.cat(labels)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return videos, flat_labels, input_lengths, label_lengths


# ============================================================================
# MODEL
# ============================================================================

class LipReadingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.backend = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, config["cnn_output_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        
        self.lstm = nn.LSTM(
            input_size=config["cnn_output_dim"],
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=config["dropout"] if config["lstm_layers"] > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(config["lstm_hidden_size"] * 2)
        self.classifier = nn.Linear(config["lstm_hidden_size"] * 2, NUM_CLASSES)
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = self.frontend(x)
        
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = self.backend(x)
        x = x.view(B, T, -1)
        
        x = self.fc(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.classifier(x)
        
        return x.permute(1, 0, 2)


# ============================================================================
# UTILITIES
# ============================================================================

def greedy_decode(log_probs):
    predictions = torch.argmax(log_probs, dim=2)
    decoded = []
    for b in range(predictions.shape[1]):
        sequence = predictions[:, b].tolist()
        result = []
        prev = -1
        for idx in sequence:
            if idx != prev and idx != BLANK_IDX:
                result.append(idx2char.get(idx, ''))
            prev = idx
        decoded.append(''.join(result))
    return decoded


def decode_labels(labels, label_lengths):
    targets = []
    idx = 0
    for length in label_lengths:
        target_indices = labels[idx:idx+length].tolist()
        target = ''.join([idx2char.get(i, '') for i in target_indices])
        targets.append(target)
        idx += length
    return targets


def compute_metrics(predictions, targets):
    if not predictions or not targets:
        return 1.0, 1.0
    if HAS_JIWER:
        total_wer, total_cer = 0, 0
        for pred, target in zip(predictions, targets):
            if target.strip():
                total_wer += wer(target, pred)
                total_cer += cer(target, pred)
        return total_wer / len(predictions), total_cer / len(predictions)
    else:
        correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        return 1 - correct / len(predictions), 1 - correct / len(predictions)


class LRScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, ctc_loss, scaler, device, config):
    model.train()
    total_loss = 0
    all_predictions, all_targets = [], []
    
    for videos, labels, input_lengths, label_lengths in tqdm(dataloader, desc="Training", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=config["use_amp"]):
            outputs = model(videos)
            log_probs = F.log_softmax(outputs, dim=2)
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            predictions = greedy_decode(log_probs.cpu())
            targets = decode_labels(labels.cpu(), label_lengths)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(dataloader)
    train_wer, train_cer = compute_metrics(all_predictions, all_targets)
    return avg_loss, train_wer, train_cer


def evaluate(model, dataloader, ctc_loss, device, config):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for videos, labels, input_lengths, label_lengths in tqdm(dataloader, desc="Evaluating", leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast('cuda', enabled=config["use_amp"]):
                outputs = model(videos)
                log_probs = F.log_softmax(outputs, dim=2)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            total_loss += loss.item()
            predictions = greedy_decode(log_probs.cpu())
            targets = decode_labels(labels.cpu(), label_lengths)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / len(dataloader)
    eval_wer, eval_cer = compute_metrics(all_predictions, all_targets)
    return avg_loss, eval_wer, eval_cer, all_predictions, all_targets


def export_model(model, config, output_dir, device):
    output_dir = Path(output_dir)
    print("\n📦 Exporting models...")
    
    with open(output_dir / "char2idx.json", 'w') as f:
        json.dump(char2idx, f)
    with open(output_dir / "idx2char.json", 'w') as f:
        json.dump({str(k): v for k, v in idx2char.items()}, f)
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print("  ✓ Saved config and mappings")
    
    model.eval()
    dummy = torch.randn(1, config["max_frames"], 1, config["frame_height"], config["frame_width"]).to(device)
    
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(str(output_dir / "model_deploy.torchscript"))
        print(f"  ✓ Saved TorchScript: {output_dir / 'model_deploy.torchscript'}")
    except Exception as e:
        print(f"  ✗ TorchScript failed: {e}")


def train(config, data_dir, output_dir):
    print("\n" + "=" * 70)
    print("🚀 FULL GRID DATASET TRAINING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    manifest_path = Path(data_dir) / "manifest.json"
    dataset = LipReadingDataset(manifest_path, config["max_frames"], 
                                (config["frame_height"], config["frame_width"]))
    
    total = len(dataset)
    val_size = int(total * config["val_split"])
    test_size = int(total * config["test_split"])
    train_size = total - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Splits: train={train_size}, val={val_size}, test={test_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                              collate_fn=collate_fn, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                            collate_fn=collate_fn, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                             collate_fn=collate_fn, persistent_workers=True)
    
    # Model
    model = LipReadingModel(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    lr_scheduler = LRScheduler(optimizer, config["warmup_epochs"], config["epochs"], config["learning_rate"])
    scaler = GradScaler('cuda', enabled=config["use_amp"])
    
    # History
    history = {"config": config, "epochs": [], 
               "train_loss": [], "train_wer": [], "train_cer": [],
               "val_loss": [], "val_wer": [], "val_cer": [], "learning_rate": []}
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config["epochs"]):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*70}")
        
        lr = lr_scheduler.step(epoch)
        print(f"LR: {lr:.6f}")
        
        train_loss, train_wer, train_cer = train_epoch(model, train_loader, optimizer, ctc_loss, scaler, device, config)
        val_loss, val_wer, val_cer, val_preds, val_targets = evaluate(model, val_loader, ctc_loss, device, config)
        
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_wer"].append(train_wer)
        history["train_cer"].append(train_cer)
        history["val_loss"].append(val_loss)
        history["val_wer"].append(val_wer)
        history["val_cer"].append(val_cer)
        history["learning_rate"].append(lr)
        
        print(f"Train - Loss: {train_loss:.4f}, WER: {train_wer:.4f}, CER: {train_cer:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, WER: {val_wer:.4f}, CER: {val_cer:.4f}")
        
        for i in range(min(2, len(val_preds))):
            print(f"  Target: '{val_targets[i]}'")
            print(f"  Pred:   '{val_preds[i]}'")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "val_wer": val_wer, "val_cer": val_cer, "config": config
            }, output_dir / "best_model.pt")
            print(f"  ✓ Saved best model")
        
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # Final test
    print(f"\n{'='*70}")
    print("📊 FINAL TEST EVALUATION")
    print(f"{'='*70}")
    
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_wer, test_cer, _, _ = evaluate(model, test_loader, ctc_loss, device, config)
    
    history["test_loss"] = test_loss
    history["test_wer"] = test_wer
    history["test_cer"] = test_cer
    history["training_time_minutes"] = (time.time() - start_time) / 60
    
    print(f"Test - Loss: {test_loss:.4f}, WER: {test_wer:.4f}, CER: {test_cer:.4f}")
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    export_model(model, config, output_dir, device)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Test CER: {test_cer:.4f}")
    print(f"Time: {history['training_time_minutes']:.1f} min")
    print(f"Models: {output_dir}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Full GRID Training")
    parser.add_argument("--data-dir", "-d", default="./processed_data_full")
    parser.add_argument("--output-dir", "-o", default="./checkpoints_full")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    train(config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
