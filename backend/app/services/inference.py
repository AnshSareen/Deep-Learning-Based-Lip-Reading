import json
import torch
import numpy as np
import cv2
from pathlib import Path
from app.config import settings
from app.utils.logger import logger
from typing import Tuple, Dict, Optional

BLANK_IDX = 0

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
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference: str, hypothesis: str) -> float:
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


def calculate_cer(reference: str, hypothesis: str) -> float:
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

TRANSCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "transcripts"


def load_transcript_from_align(video_path: str) -> Optional[str]:
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


def parse_align_file(align_path: Path) -> str:
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

class LipReadingInference:
    def __init__(self):  # <-- FIX THIS (was __init**)
        self.device = self._setup_device()
        self.model = None
        self.idx2char = None
        self.load_model()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if settings.USE_GPU and torch.cuda.is_available():
            device = torch.device(f"cuda:{settings.GPU_ID}")
            logger.info(f"✓ Using GPU: {torch.cuda.get_device_name(settings.GPU_ID)}")
        else:
            device = torch.device("cpu")
            logger.info("✓ Using CPU")
        return device
    
    def load_model(self) -> None:
        """Load TorchScript model"""
        try:
            model_path = settings.MODEL_PATH
            
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                logger.warning("Model will not be loaded. API will work but inference will fail.")
                self.model = None
                return

            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
            logger.info(f"Loaded model from: {model_path}")
            
            # Load character mapping
            if not settings.IDX2CHAR_PATH.exists():
                logger.warning(f"⚠️  Character mapping not found: {settings.IDX2CHAR_PATH}")
                self.idx2char = {}
                return
                
            with open(settings.IDX2CHAR_PATH, 'r') as f:
                mapping = json.load(f)
            self.idx2char = {int(k): v for k, v in mapping.items()}
            logger.info("✓ Loaded character mapping")
            
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            self.model = None
            self.idx2char = {}
    
    def preprocess_frames(self, frames: np.ndarray) -> Tuple[torch.Tensor, int]:
        """Preprocess frames for model input"""
        try:
            num_frames = len(frames)
            
            # Resize frames
            resized_frames = []
            for frame in frames:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - 0.5) / 0.5  # Normalize
                resized_frames.append(frame)
            
            # Pad to max frames
            while len(resized_frames) < settings.MAX_FRAMES:
                resized_frames.append(resized_frames[-1])
            
            # Create video tensor: [1, T, 1, H, W]
            video = np.stack(resized_frames[:settings.MAX_FRAMES])
            video = np.expand_dims(video, axis=(0, 2))
            
            return torch.from_numpy(video).float(), num_frames
        
        except Exception as e:
            logger.error(f"✗ Frame preprocessing failed: {e}")
            raise
    
    def greedy_decode(self, log_probs: torch.Tensor) -> Tuple[str, float]:
        """CTC greedy decoding with confidence"""
        try:
            predictions = torch.argmax(log_probs, dim=2)
            probabilities = torch.softmax(log_probs, dim=2)
            
            sequence = predictions[:, 0].tolist()
            probs = probabilities[:, 0]
            
            result = []
            confidence_scores = []
            prev = -1
            
            for idx, prob in zip(sequence, probs):
                if idx != prev and idx != BLANK_IDX:
                    result.append(self.idx2char.get(idx, ''))
                    confidence_scores.append(float(prob[idx].item()))
                prev = idx
            
            text = ''.join(result)
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return text, avg_confidence
        
        except Exception as e:
            logger.error(f"✗ Decoding failed: {e}")
            raise
    
    def infer(self, frames: np.ndarray, video_path: str = None) -> Dict:
        """Run inference on frames"""
        import time
        
        try:
            if self.model is None:
                return {
                    "success": False,
                    "error": "Model not loaded. Please ensure model files are present in app/ml_models/"
                }
            
            start_time = time.time()
            
            # Preprocess
            video_tensor, num_frames = self.preprocess_frames(frames)
            
            # Inference
            with torch.no_grad():
                video_tensor = video_tensor.to(self.device)
                outputs = self.model(video_tensor)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                
                text, confidence = self.greedy_decode(log_probs.cpu())
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "predicted_text": text,
                "confidence_score": confidence,
                "frames_detected": num_frames,
                "processing_time": processing_time,
                "ground_truth": None,
                "word_error_rate": None,
                "char_error_rate": None,
                "is_exact_match": None
            }
            
            # Calculate WER/CER if video_path is provided
            if video_path:
                ground_truth = load_transcript_from_align(video_path)
                if ground_truth:
                    result["ground_truth"] = ground_truth
                    result["word_error_rate"] = calculate_wer(ground_truth, text)
                    result["char_error_rate"] = calculate_cer(ground_truth, text)
                    result["is_exact_match"] = text.strip().lower() == ground_truth.strip().lower()
                    logger.info(f"✓ WER: {result['word_error_rate']*100:.2f}%, CER: {result['char_error_rate']*100:.2f}%")
            
            return result
        
        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global inference instance
_inference_instance = None

def get_inference_engine() -> LipReadingInference:
    """Get or create inference engine"""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = LipReadingInference()
    return _inference_instance