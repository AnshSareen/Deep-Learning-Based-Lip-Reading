from sqlalchemy import Column, String, DateTime, Float, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class VideoAnalysis(Base):
    __tablename__ = "video_analyses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Analysis Results
    predicted_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    frames_detected = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Status
    status = Column(String(50), default="pending")
    error_message = Column(Text, nullable=True)
    
    # Metadata
    video_duration = Column(Float, nullable=True)
    frame_rate = Column(Float, nullable=True)
    resolution = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<VideoAnalysis(id={self.id}, filename={self.filename}, status={self.status})>"

class ProcessingLog(Base):
    __tablename__ = "processing_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String(36), nullable=False)
    step = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessingLog(analysis_id={self.analysis_id}, step={self.step})>"