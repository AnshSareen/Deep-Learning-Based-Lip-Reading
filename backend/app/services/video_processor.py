import cv2
import numpy as np
from pathlib import Path
from app.utils.logger import logger
from typing import Tuple, Optional
import os

class VideoProcessor:
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "resolution": f"{width}x{height}"
            }
        
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    @staticmethod
    def extract_frames(video_path: str, target_fps: int = 25) -> Tuple[list, dict]:
        """Extract frames from video"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame sampling rate
            sample_rate = max(1, int(fps / target_fps))
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % sample_rate == 0:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            info = {
                "total_frames": frame_count,
                "extracted_frames": extracted_count,
                "original_fps": fps,
                "target_fps": target_fps,
                "sample_rate": sample_rate
            }
            
            logger.info(f"✓ Extracted {extracted_count} frames from {frame_count} total")
            return frames, info
        
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return [], {}
    
    @staticmethod
    def validate_video(video_path: str, max_size: int = None) -> Tuple[bool, str]:
        """Validate video file"""
        try:
            path = Path(video_path)
            
            # Check file exists
            if not path.exists():
                return False, "Video file not found"
            
            # Check file size
            if max_size and path.stat().st_size > max_size:
                return False, f"File size exceeds {max_size} bytes"
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if frame_count == 0:
                return False, "Video has no frames"
            
            return True, "Valid"
        
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def compress_video(input_path: str, output_path: str, quality: int = 85) -> bool:
        """Compress video for faster processing"""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Reduce quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, frame = cv2.imencode('.jpg', frame, encode_param)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                
                out.write(frame)
            
            cap.release()
            out.release()
            
            logger.info(f"✓ Compressed video saved to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error compressing video: {e}")
            return False

def get_video_processor() -> VideoProcessor:
    """Get video processor instance"""
    return VideoProcessor()