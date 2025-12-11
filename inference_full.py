#!/usr/bin/env python3
"""
Full Model Inference Script
===========================
Run inference using the fully trained (30 epochs, 34k videos) model.
Supports inputting a video file and getting prediction vs actual with WER/CER metrics.

Usage:
    python inference_full.py --video path/to/video.mpg
    python inference_full.py --video path/to/video.mpg --transcript "override text"
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_FRAMES = 75
FRAME_HEIGHT = 50
FRAME_WIDTH = 100
BLANK_IDX = 0

# Load character mappings from the current directory
CHECKPOINT_DIR = Path(".")
TRANSCRIPTS_DIR = CHECKPOINT_DIR / "transcripts"

try:
    with open(CHECKPOINT_DIR / "idx2char.json", "r") as f:
        idx2char = json.load(f)
        # Convert keys to integers
        idx2char = {int(k): v for k, v in idx2char.items()}
except FileNotFoundError:
    print("Warning: idx2char.json not found. Using default.")
    CHARSET = "abcdefghijklmnopqrstuvwxyz '"
    idx2char = {idx + 1: ch for idx, ch in enumerate(CHARSET)}


# ============================================================================
# WER & CER CALCULATION
# ============================================================================

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER).
    WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=words in reference
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    return wer


def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=chars in reference
    """
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())
    
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    return cer


# ============================================================================
# TRANSCRIPT LOOKUP
# ============================================================================

def load_transcript_from_align(video_path):
    """
    Load the ground truth transcript from the transcripts folder.
    Searches through all speaker folders (s1-s34) to find the matching .align file.
    """
    video_name = Path(video_path).stem  # e.g., "bbas1s"
    align_filename = f"{video_name}.align"
    
    # Search through all speaker folders
    if TRANSCRIPTS_DIR.exists():
        for speaker_dir in TRANSCRIPTS_DIR.iterdir():
            if speaker_dir.is_dir():
                align_path = speaker_dir / "align" / align_filename
                if align_path.exists():
                    return parse_align_file(align_path)
    
    return None


def parse_align_file(align_path):
    """
    Parse a .align file and extract the transcript.
    Format: <start_time> <end_time> <word>
    Skip 'sil' (silence) entries.
    """
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word = parts[2]
                if word.lower() != 'sil':  # Skip silence markers
                    words.append(word.lower())
    
    return ' '.join(words)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_full_model(device):
    """Load the fully trained TorchScript model."""
    # Use the zip file which torch.jit.load can read directly
    model_path = CHECKPOINT_DIR / "model_deploy.torchscript.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model

# ============================================================================
# PREPROCESSING
# ============================================================================

def load_and_process_video(video_path):
    """
    Load a video file, detect face using Haar Cascade, crop lips using EXACT training logic,
    and preprocess for the model.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    # Load face detector (same as training)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    padding = 0 # Training script had padding=0 default in crop_lips function
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # EXACT logic from preprocess_grid.py:
            # lip_y_start = y + int(fh * 0.65)
            # lip_y_end = y + int(fh * 0.95)
            # lip_x_start = x + int(fw * 0.20)
            # lip_x_end = x + int(fw * 0.80)
            
            lip_y_start = y + int(h * 0.65)
            lip_y_end = y + int(h * 0.95)
            lip_x_start = x + int(w * 0.20)
            lip_x_end = x + int(w * 0.80)
            
            # Add padding (if any)
            lip_y_start = max(0, lip_y_start - padding)
            lip_y_end = min(gray.shape[0], lip_y_end + padding)
            lip_x_start = max(0, lip_x_start - padding)
            lip_x_end = min(gray.shape[1], lip_x_end + padding)
            
            # Validate crop
            if lip_x_end > lip_x_start and lip_y_end > lip_y_start:
                crop = gray[lip_y_start:lip_y_end, lip_x_start:lip_x_end]
                processed_frame = cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT))
            else:
                # Fallback to center crop if calc fails
                h_img, w_img = gray.shape
                cy, cx = h_img // 2, w_img // 2
                crop = gray[cy-25:cy+25, cx-50:cx+50]
                processed_frame = cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT))
                
        else:
            # Fallback if no face detected
            h_img, w_img = gray.shape
            cy, cx = h_img // 2, w_img // 2
            crop = gray[cy-25:cy+25, cx-50:cx+50] 
            processed_frame = cv2.resize(crop, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Normalize
        processed_frame = processed_frame.astype(np.float32) / 255.0
        processed_frame = (processed_frame - 0.5) / 0.5
        
        frames.append(processed_frame)
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames extracted from video")

    # Pad or truncate to MAX_FRAMES
    if len(frames) > MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    else:
        while len(frames) < MAX_FRAMES:
            frames.append(frames[-1]) # Pad with last frame
            
    # Stack and add dimensions: [1, T, 1, H, W]
    video_tensor = np.stack(frames)
    video_tensor = np.expand_dims(video_tensor, axis=(0, 2))
    video_tensor = torch.from_numpy(video_tensor).float()
    
    return video_tensor

# ============================================================================
# INFERENCE
# ============================================================================

def greedy_decode(log_probs):
    """CTC greedy decoding."""
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
    
    return decoded[0]

def predict(model, video_path, device):
    """Run inference on a single video file."""
    try:
        video_tensor = load_and_process_video(video_path)
    except Exception as e:
        return f"Error processing video: {e}"

    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        outputs = model(video_tensor)
        log_probs = F.log_softmax(outputs, dim=2)
        prediction = greedy_decode(log_probs.cpu())
    
    return prediction

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run inference with full lip reading model")
    parser.add_argument("--video", "-v", required=True, help="Path to input video file (e.g., demo_videos/bbas1s.mpg)")
    parser.add_argument("--transcript", "-t", default=None, help="Override transcript (optional). If not provided, auto-lookup from transcripts folder.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = load_full_model(device)
        prediction = predict(model, args.video, device)
        
        # Get transcript - either from argument or auto-lookup
        if args.transcript:
            transcript = args.transcript
            transcript_source = "provided by user"
        else:
            transcript = load_transcript_from_align(args.video)
            transcript_source = "auto-loaded from transcripts folder"
        
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Video File: {args.video}")
        print(f"Model Prediction: '{prediction}'")
        
        if transcript:
            print(f"Ground Truth: '{transcript}' ({transcript_source})")
            
            # Calculate metrics
            wer = calculate_wer(transcript, prediction)
            cer = calculate_cer(transcript, prediction)
            
            print("-"*60)
            print("METRICS")
            print("-"*60)
            print(f"Word Error Rate (WER):      {wer*100:.2f}%")
            print(f"Character Error Rate (CER): {cer*100:.2f}%")
            
            if prediction.strip().lower() == transcript.strip().lower():
                print("\nResult: EXACT MATCH ✅")
            else:
                print("\nResult: MISMATCH ❌")
        else:
            print("Ground Truth: Not found (no transcript available)")
            
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
