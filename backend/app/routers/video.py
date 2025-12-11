import asyncio
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

from app.config import settings
from app.schemas import VideoUploadResponse, VideoAnalysisResponse, InferenceResult
from app.services.video_processor import get_video_processor
from app.services.lip_extractor import get_lip_extractor
from app.services.inference import get_inference_engine
from app.utils.logger import logger
from app.utils.helpers import generate_unique_filename, validate_file_extension, ensure_directory_exists

router = APIRouter(prefix="/api/videos", tags=["videos"])

video_processor = get_video_processor()
lip_extractor = get_lip_extractor()
inference_engine = get_inference_engine()

# Ensure upload directories exist
ensure_directory_exists(settings.UPLOAD_DIR / "videos")
ensure_directory_exists(settings.UPLOAD_DIR / "frames")

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video"""
    try:
        # Validate file
        if not validate_file_extension(file.filename, settings.ALLOWED_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Save uploaded file
        unique_filename = generate_unique_filename(file.filename, "video")
        upload_path = settings.UPLOAD_DIR / "videos" / unique_filename
        
        contents = await file.read()
        with open(upload_path, 'wb') as f:
            f.write(contents)
        
        logger.info(f"✓ Video uploaded: {unique_filename}")
        
        # Validate video
        is_valid, validation_msg = video_processor.validate_video(str(upload_path), settings.MAX_UPLOAD_SIZE)
        if not is_valid:
            upload_path.unlink()
            raise HTTPException(status_code=400, detail=validation_msg)
        
        # Create analysis record (would save to DB)
        analysis_id = f"analysis_{int(time.time())}_{unique_filename}"
        
        return VideoUploadResponse(
            analysis_id=analysis_id,
            message="Video uploaded successfully. Processing started.",
            status="processing"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/{analysis_id}", response_model=InferenceResult)
async def analyze_video(analysis_id: str, background_tasks: BackgroundTasks):
    """Analyze uploaded video"""
    try:
        # Find video file (in production, would query database)
        videos_dir = settings.UPLOAD_DIR / "videos"
        video_files = list(videos_dir.glob(f"*{analysis_id.split('_')[-1]}"))
        
        if not video_files:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_path = str(video_files[0])
        
        logger.info(f"Starting analysis of {video_path}")
        
        # Extract frames
        frames, frame_info = video_processor.extract_frames(video_path, settings.FPS_TARGET)
        
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
        
        # Extract lip regions
        lip_frames = lip_extractor.extract_lip_frames(video_path, sample_rate=2)
        
        if not lip_frames:
            # Fallback: use extracted frames
            lip_frames = frames
        
        # Convert to numpy array
        frames_array = np.array(lip_frames[:settings.MAX_FRAMES])
        
        # Run inference with video path for WER/CER calculation
        result = inference_engine.infer(frames_array, video_path=video_path)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
        
        logger.info(f"✓ Analysis complete: {result['predicted_text']}")
        
        return InferenceResult(
            analysis_id=analysis_id,
            predicted_text=result["predicted_text"],
            confidence_score=result["confidence_score"],
            frames_detected=result["frames_detected"],
            processing_time=result["processing_time"],
            success=True,
            ground_truth=result.get("ground_truth"),
            word_error_rate=result.get("word_error_rate"),
            char_error_rate=result.get("char_error_rate"),
            is_exact_match=result.get("is_exact_match")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_analysis_history(limit: int = 10, offset: int = 0):
    """Get analysis history"""
    try:
        # In production, would query database
        return {
            "total": 0,
            "analyses": []
        }
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status"""
    try:
        # In production, would query database
        return {
            "analysis_id": analysis_id,
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Demo Video Endpoints
DEMO_VIDEOS_PATH = Path(__file__).parent.parent.parent.parent / "demo_videos"

@router.get("/demo")
async def list_demo_videos():
    """List available demo videos for testing"""
    try:
        if not DEMO_VIDEOS_PATH.exists():
            return {"demo_videos": [], "message": "Demo videos folder not found"}
        
        videos = list(DEMO_VIDEOS_PATH.glob("*.mpg")) + list(DEMO_VIDEOS_PATH.glob("*.mp4"))
        return {
            "demo_videos": [v.name for v in videos],
            "path": str(DEMO_VIDEOS_PATH),
            "count": len(videos)
        }
    except Exception as e:
        logger.error(f"Demo videos list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demo/{video_name}", response_model=InferenceResult)
async def analyze_demo_video(video_name: str):
    """Run inference directly on a demo video"""
    try:
        video_path = DEMO_VIDEOS_PATH / video_name
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Demo video '{video_name}' not found")
        
        logger.info(f"🎬 Analyzing demo video: {video_name}")
        
        # Extract lip frames
        lip_frames = lip_extractor.extract_lip_frames(str(video_path), sample_rate=2)
        
        if not lip_frames:
            # Fallback: extract frames directly
            frames, _ = video_processor.extract_frames(str(video_path), settings.FPS_TARGET)
            lip_frames = frames
        
        if not lip_frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
        
        # Convert to numpy array
        frames_array = np.array(lip_frames[:settings.MAX_FRAMES])
        
        # Run inference with video path for WER/CER calculation
        result = inference_engine.infer(frames_array, video_path=str(video_path))
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Inference failed"))
        
        logger.info(f"✓ Demo analysis complete: {result['predicted_text']}")
        
        return InferenceResult(
            analysis_id=f"demo_{video_name}_{int(time.time())}",
            predicted_text=result["predicted_text"],
            confidence_score=result["confidence_score"],
            frames_detected=result["frames_detected"],
            processing_time=result["processing_time"],
            success=True,
            ground_truth=result.get("ground_truth"),
            word_error_rate=result.get("word_error_rate"),
            char_error_rate=result.get("char_error_rate"),
            is_exact_match=result.get("is_exact_match")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))