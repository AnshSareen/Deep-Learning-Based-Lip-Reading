import torch
from fastapi import APIRouter, HTTPException
from app.schemas import HealthResponse
from app.config import settings
from app.services.inference import get_inference_engine
from app.utils.logger import logger

router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check model
        inference = get_inference_engine()
        model_loaded = inference.model is not None
        
        # Check GPU
        gpu_available = torch.cuda.is_available()
        
        return HealthResponse(
            status="healthy",
            version=settings.API_VERSION,
            model_loaded=model_loaded,
            gpu_available=gpu_available
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ready")
async def readiness_check():
    """Readiness probe"""
    try:
        inference = get_inference_engine()
        if inference.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_check():
    """Liveness probe"""
    return {"status": "alive"}