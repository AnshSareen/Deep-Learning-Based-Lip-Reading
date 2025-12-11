"""
Lip Extractor using Haar Cascade
================================
Uses the same preprocessing logic as inference_full.py for consistent predictions.
"""
import cv2
import numpy as np
from app.utils.logger import logger
from app.config import settings
from typing import List, Tuple, Optional

class LipExtractor:
    def __init__(self):
        # Load Haar Cascade face detector (same as inference_full.py)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.padding = 0  # Same as training script
    
    def extract_lip_region_from_gray(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract lip region from grayscale frame using Haar Cascade.
        Uses EXACT logic from inference_full.py and preprocess_grid.py:
        - lip_y_start = y + int(h * 0.65)
        - lip_y_end = y + int(h * 0.95)
        - lip_x_start = x + int(w * 0.20)
        - lip_x_end = x + int(w * 0.80)
        """
        try:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # EXACT logic from preprocess_grid.py
                lip_y_start = y + int(h * 0.65)
                lip_y_end = y + int(h * 0.95)
                lip_x_start = x + int(w * 0.20)
                lip_x_end = x + int(w * 0.80)
                
                # Add padding (if any)
                lip_y_start = max(0, lip_y_start - self.padding)
                lip_y_end = min(gray.shape[0], lip_y_end + self.padding)
                lip_x_start = max(0, lip_x_start - self.padding)
                lip_x_end = min(gray.shape[1], lip_x_end + self.padding)
                
                # Validate crop
                if lip_x_end > lip_x_start and lip_y_end > lip_y_start:
                    crop = gray[lip_y_start:lip_y_end, lip_x_start:lip_x_end]
                    return crop
                else:
                    # Fallback to center crop if calc fails
                    return self._center_crop(gray)
            else:
                # Fallback if no face detected
                return self._center_crop(gray)
                
        except Exception as e:
            logger.error(f"Error extracting lip region: {e}")
            return None
    
    def _center_crop(self, gray: np.ndarray) -> np.ndarray:
        """Fallback center crop when face detection fails."""
        h_img, w_img = gray.shape
        cy, cx = h_img // 2, w_img // 2
        # Use safe crop boundaries
        y1 = max(0, cy - 25)
        y2 = min(h_img, cy + 25)
        x1 = max(0, cx - 50)
        x2 = min(w_img, cx + 50)
        return gray[y1:y2, x1:x2]
    
    def extract_lip_frames(self, video_path: str, sample_rate: int = 1,
                          target_size: Tuple[int, int] = (100, 50)) -> List[np.ndarray]:
        """
        Extract lip frames from video using Haar Cascade.
        Uses same preprocessing as inference_full.py for consistent predictions.
        """
        lip_frames = []

        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            detected_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale first (same as inference_full.py)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract lip region using Haar Cascade
                lip_region = self.extract_lip_region_from_gray(gray)
                
                if lip_region is not None and lip_region.size > 0:
                    # Resize to target size (FRAME_WIDTH=100, FRAME_HEIGHT=50)
                    lip_region_resized = cv2.resize(lip_region, target_size)
                    lip_frames.append(lip_region_resized)
                    detected_count += 1

                frame_count += 1

            cap.release()
            logger.info(f"Extracted {detected_count} lip frames from {frame_count} total frames")

            return lip_frames

        except Exception as e:
            logger.error(f"Error extracting lip frames: {e}")
            return []

def get_lip_extractor() -> LipExtractor:
    """Get lip extractor instance"""
    return LipExtractor()