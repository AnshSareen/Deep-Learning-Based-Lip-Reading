#!/usr/bin/env python3
"""
GRID Lip Reading - Frame Extraction & Preprocessing (Parallel Version)
========================================================================
Extracts frames from MPG videos and crops lip regions for training.
Optimized for high-core-count CPUs (e.g., 128 threads).

Features:
- Thread-safe parallel processing with ProcessPoolExecutor
- No race conditions (each worker processes independent videos)
- Per-worker face cascade (no shared state)
- Atomic file operations with markers

Usage:
    python preprocess_grid.py --data-dir ./grid_dataset --output-dir ./processed_data --workers 32

Requirements:
    - OpenCV (pip install opencv-python-headless)
    - tqdm (pip install tqdm)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output frame size
OUTPUT_HEIGHT = 50
OUTPUT_WIDTH = 100

# Default number of workers (will be overridden by system check)
DEFAULT_WORKERS = 32

# Haar cascade path (will be loaded per-worker to avoid sharing issues)
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# ============================================================================
# WORKER FUNCTIONS (Each runs in separate process - no shared state)
# ============================================================================

def init_worker():
    """
    Initialize worker process with its own face cascade.
    Called once when each worker process starts.
    """
    global worker_face_cascade
    worker_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def crop_lip_region_worker(image, padding=10):
    """
    Crop the lip region from an image using the worker's face cascade.
    Thread-safe: uses per-worker face cascade.
    
    Args:
        image: BGR image (numpy array)
        padding: Pixels to pad around lip region
    
    Returns:
        Cropped lip region or None if face not detected
    """
    global worker_face_cascade
    
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using worker's cascade
    faces = worker_face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        # Fallback: use center-bottom region (works for GRID frontal faces)
        lip_region = image[int(h*0.55):int(h*0.85), int(w*0.25):int(w*0.75)]
        if lip_region.size > 0:
            return cv2.resize(lip_region, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        return None
    
    # Use the first (largest) face
    x, y, fw, fh = faces[0]
    
    # Lip region is in the lower third of the face
    lip_y_start = y + int(fh * 0.65)
    lip_y_end = y + int(fh * 0.95)
    lip_x_start = x + int(fw * 0.20)
    lip_x_end = x + int(fw * 0.80)
    
    # Add padding
    lip_y_start = max(0, lip_y_start - padding)
    lip_y_end = min(h, lip_y_end + padding)
    lip_x_start = max(0, lip_x_start - padding)
    lip_x_end = min(w, lip_x_end + padding)
    
    if lip_x_end <= lip_x_start or lip_y_end <= lip_y_start:
        return None
    
    lip_crop = image[lip_y_start:lip_y_end, lip_x_start:lip_x_end]
    
    if lip_crop.size == 0:
        return None
    
    # Resize to standard size
    return cv2.resize(lip_crop, (OUTPUT_WIDTH, OUTPUT_HEIGHT))


def process_single_video(args):
    """
    Process a single video: extract frames and crop lips.
    
    This function is completely self-contained and thread-safe.
    Each call operates on independent files with no shared state.
    
    Args:
        args: tuple of (video_path, frames_dir, lips_dir, speaker_id)
    
    Returns:
        dict with processing statistics
    """
    video_path, frames_base_dir, lips_base_dir, speaker_id = args
    video_path = Path(video_path)
    video_id = video_path.stem
    
    # Create unique output directories for this video (no collisions)
    frames_output = Path(frames_base_dir) / speaker_id / video_id
    lips_output = Path(lips_base_dir) / speaker_id / video_id
    
    result = {
        "video_id": video_id,
        "speaker_id": speaker_id,
        "frames_extracted": 0,
        "lips_cropped": 0,
        "status": "success",
        "error": None
    }
    
    try:
        # ===== STEP 1: Extract frames (if not already done) =====
        frames_marker = frames_output / ".frames_done"
        
        if frames_marker.exists():
            # Already extracted, count existing frames
            result["frames_extracted"] = len(list(frames_output.glob("*.jpg")))
        else:
            # Create directory (atomic - ok if exists)
            frames_output.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result["status"] = "error"
                result["error"] = "Could not open video"
                return result
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_path = frames_output / f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
            
            cap.release()
            result["frames_extracted"] = frame_count
            
            # Mark as complete (atomic)
            frames_marker.touch()
        
        # ===== STEP 2: Crop lip regions (if not already done) =====
        lips_marker = lips_output / ".lips_done"
        
        if lips_marker.exists():
            # Already cropped, count existing
            result["lips_cropped"] = len(list(lips_output.glob("*.jpg")))
        else:
            # Create directory
            lips_output.mkdir(parents=True, exist_ok=True)
            
            # Initialize face cascade for this worker
            init_worker()
            
            # Process each frame
            lips_cropped = 0
            for frame_file in sorted(frames_output.glob("*.jpg")):
                image = cv2.imread(str(frame_file))
                lip_crop = crop_lip_region_worker(image)
                
                if lip_crop is not None:
                    # Use same filename for easy matching
                    lip_path = lips_output / frame_file.name
                    cv2.imwrite(str(lip_path), lip_crop)
                    lips_cropped += 1
            
            result["lips_cropped"] = lips_cropped
            
            # Mark as complete (atomic)
            if lips_cropped > 0:
                lips_marker.touch()
        
        return result
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


# ============================================================================
# ALIGNMENT PROCESSING
# ============================================================================

def parse_alignment_file(align_path):
    """Parse a GRID alignment file."""
    with open(align_path, 'r') as f:
        lines = f.readlines()
    
    words = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            word = parts[2].lower()
            if word != 'sil':
                words.append(word)
    
    return ' '.join(words)


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_dataset(data_dir, output_dir, speakers=None, max_workers=None):
    """
    Main preprocessing pipeline with parallel execution.
    
    Args:
        data_dir: Directory containing downloaded GRID data
        output_dir: Directory to save processed data
        speakers: List of speakers to process (e.g., ['s1', 's2', 's3'])
        max_workers: Number of parallel workers
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Auto-detect optimal workers based on CPU
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        # Use 50% of available cores (leave room for I/O)
        max_workers = max(1, cpu_count // 2)
    
    # Cap at reasonable maximum to avoid memory issues
    max_workers = min(max_workers, 64)
    
    # Directories
    frames_dir = output_dir / "frames"
    lips_dir = output_dir / "cropped_lips"
    
    print("\n" + "=" * 70)
    print("GRID DATASET PREPROCESSING (PARALLEL)")
    print("=" * 70)
    print(f"Input directory:  {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"CPU cores:        {cpu_count}")
    print(f"Workers:          {max_workers}")
    
    # Find video directories
    video_base = data_dir / "video"
    
    if speakers is None:
        speakers = []
        if video_base.exists():
            for speaker_dir in sorted(video_base.iterdir()):
                if speaker_dir.is_dir():
                    speakers.append(speaker_dir.name)
    
    print(f"Speakers:         {speakers}")
    
    # Collect all video files with their metadata
    video_tasks = []
    for speaker in speakers:
        # Skip non-speaker directories
        if not speaker.startswith('s') or speaker.startswith('__'):
            continue
        
        # Try different possible directory structures
        possible_dirs = [
            video_base / speaker / speaker,  # video/s1/s1/*.mpg
            video_base / speaker,             # video/s1/*.mpg
        ]
        
        speaker_video_dir = None
        for d in possible_dirs:
            if d.exists() and list(d.glob("*.mpg")):
                speaker_video_dir = d
                break
        
        if speaker_video_dir:
            for video_file in speaker_video_dir.glob("*.mpg"):
                video_tasks.append((
                    str(video_file),  # video_path
                    str(frames_dir),  # frames_base_dir
                    str(lips_dir),    # lips_base_dir
                    speaker           # speaker_id
                ))
    
    print(f"Total videos:     {len(video_tasks)}")
    
    if not video_tasks:
        print("\nNo video files found!")
        return
    
    # Process videos in parallel
    print("\n" + "=" * 70)
    print("PROCESSING VIDEOS")
    print("=" * 70)
    
    results = []
    errors = []
    
    # Use ProcessPoolExecutor for true parallel processing
    # Each process has its own memory space - NO race conditions
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_video, task): task for task in video_tasks}
        
        # Process results as they complete
        with tqdm(total=len(video_tasks), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "error":
                        errors.append(f"{result['video_id']}: {result['error']}")
                    
                except Exception as e:
                    task = futures[future]
                    errors.append(f"Task failed: {task[0]} - {str(e)}")
                
                pbar.update(1)
    
    # Calculate totals
    total_frames = sum(r["frames_extracted"] for r in results)
    total_lips = sum(r["lips_cropped"] for r in results)
    successful = sum(1 for r in results if r["status"] == "success")
    
    print(f"\nProcessing complete!")
    print(f"  Successful:       {successful}/{len(video_tasks)}")
    print(f"  Total frames:     {total_frames:,}")
    print(f"  Total lip crops:  {total_lips:,}")
    if errors:
        print(f"  Errors:           {len(errors)}")
        for e in errors[:5]:  # Show first 5 errors
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")
    
    # Create manifest
    print("\n" + "=" * 70)
    print("CREATING DATASET MANIFEST")
    print("=" * 70)
    
    manifest_path = output_dir / "manifest.json"
    manifest = []
    align_base = data_dir / "alignments" / "alignments"
    
    for speaker in speakers:
        speaker_lips_dir = lips_dir / speaker
        if not speaker_lips_dir.exists():
            continue
        
        for video_dir in sorted(speaker_lips_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            
            video_id = video_dir.name
            
            # Find alignment file
            align_file = align_base / speaker / f"{video_id}.align"
            if not align_file.exists():
                continue
            
            # Count frames
            frame_files = sorted(video_dir.glob("*.jpg"))
            if len(frame_files) < 10:
                continue
            
            transcript = parse_alignment_file(align_file)
            
            manifest.append({
                "video_id": video_id,
                "speaker_id": speaker,
                "lips_dir": str(video_dir),
                "num_frames": len(frame_files),
                "transcript": transcript,
                "transcript_length": len(transcript)
            })
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest created: {manifest_path}")
    print(f"Total samples:    {len(manifest)}")
    
    # Summary statistics
    if manifest:
        frame_counts = [m["num_frames"] for m in manifest]
        transcript_lengths = [m["transcript_length"] for m in manifest]
        
        print(f"\nDataset Statistics:")
        print(f"  Frame counts:       min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.1f}")
        print(f"  Transcript lengths: min={min(transcript_lengths)}, max={max(transcript_lengths)}, mean={np.mean(transcript_lengths):.1f}")
    
    return manifest


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GRID dataset for lip reading (parallel version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 32 workers (recommended for 128-thread CPU)
  python preprocess_grid.py --workers 32

  # Auto-detect optimal workers
  python preprocess_grid.py --workers auto
  
  # Process specific speakers only
  python preprocess_grid.py --speakers s1 s2 s3 --workers 16
        """
    )
    
    parser.add_argument("--data-dir", "-d", default="./grid_dataset", 
                        help="Directory containing downloaded GRID data")
    parser.add_argument("--output-dir", "-o", default="./processed_data",
                        help="Directory to save processed data")
    parser.add_argument("--speakers", nargs="+", default=None,
                        help="Specific speakers to process (e.g., --speakers s1 s2 s3)")
    parser.add_argument("--workers", type=str, default="auto",
                        help="Number of parallel workers (default: auto = 50%% of cores)")
    
    args = parser.parse_args()
    
    # Parse workers argument
    if args.workers == "auto":
        max_workers = None  # Will be auto-detected
    else:
        try:
            max_workers = int(args.workers)
        except ValueError:
            print(f"Invalid workers value: {args.workers}. Using auto.")
            max_workers = None
    
    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        speakers=args.speakers,
        max_workers=max_workers
    )


if __name__ == "__main__":
    main()
