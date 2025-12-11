"""
Metrics utilities for WER and CER calculation.
Matches logic from inference_full.py for consistent results.
"""
from pathlib import Path
from typing import Optional, Tuple

# Transcripts directory
TRANSCRIPTS_DIR = Path("../transcripts")

def levenshtein_distance(s1, s2) -> int:
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
    return min(wer, 1.0)  # Cap at 100%


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
    return min(cer, 1.0)  # Cap at 100%


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


def load_ground_truth(video_name: str) -> Optional[str]:
    """
    Load the ground truth transcript from the transcripts folder.
    Searches through all speaker folders (s1-s34) to find the matching .align file.
    """
    # Remove extension from video name
    video_stem = Path(video_name).stem  # e.g., "bbas1s"
    align_filename = f"{video_stem}.align"
    
    # Search through all speaker folders
    if TRANSCRIPTS_DIR.exists():
        for speaker_dir in TRANSCRIPTS_DIR.iterdir():
            if speaker_dir.is_dir():
                align_path = speaker_dir / "align" / align_filename
                if align_path.exists():
                    return parse_align_file(align_path)
    
    return None


def calculate_metrics(prediction: str, video_name: str) -> Tuple[Optional[str], Optional[float], Optional[float], bool]:
    """
    Calculate WER and CER metrics for a prediction.
    Returns: (ground_truth, wer, cer, is_exact_match)
    """
    ground_truth = load_ground_truth(video_name)
    
    if ground_truth is None:
        return None, None, None, False
    
    wer = calculate_wer(ground_truth, prediction)
    cer = calculate_cer(ground_truth, prediction)
    is_match = prediction.strip().lower() == ground_truth.strip().lower()
    
    return ground_truth, wer, cer, is_match
