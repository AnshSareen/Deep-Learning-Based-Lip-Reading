from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoAnalysisBase(BaseModel):
    filename: str
    original_filename: str

class VideoAnalysisCreate(VideoAnalysisBase):
    pass

class VideoAnalysisResponse(VideoAnalysisBase):
    id: str
    status: StatusEnum
    predicted_text: Optional[str] = None
    confidence_score: Optional[float] = None
    frames_detected: Optional[int] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    video_duration: Optional[float] = None
    frame_rate: Optional[float] = None
    resolution: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class VideoUploadResponse(BaseModel):
    analysis_id: str
    message: str
    status: str

class InferenceResult(BaseModel):
    analysis_id: str
    predicted_text: str
    confidence_score: float
    frames_detected: int
    processing_time: float
    success: bool
    # WER/CER metrics
    ground_truth: Optional[str] = None
    word_error_rate: Optional[float] = None
    char_error_rate: Optional[float] = None
    is_exact_match: Optional[bool] = None

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    version: str
    model_loaded: bool
    gpu_available: bool

class AnalysisHistoryResponse(BaseModel):
    total: int
    analyses: List[VideoAnalysisResponse]

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    timestamp: datetime