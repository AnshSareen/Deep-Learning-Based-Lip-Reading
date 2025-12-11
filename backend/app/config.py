from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Configuration
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_TITLE: str = "Lip Reading Analysis API"
    API_VERSION: str = "1.0.0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # File Upload
    MAX_UPLOAD_SIZE: int = 100000000
    UPLOAD_DIR: Path = Path("./uploads")
    ALLOWED_EXTENSIONS: Union[List[str], str] = ["mp4", "avi", "mov", "mkv", "flv", "wmv", "mpg", "mpeg"]

    @field_validator('ALLOWED_EXTENSIONS', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',')]
        return v
    
    # Model Configuration
    MODEL_PATH: Path = Path("./app/ml_models/model.zip")
    CHAR2IDX_PATH: Path = Path("./app/ml_models/char2idx.json")
    IDX2CHAR_PATH: Path = Path("./app/ml_models/idx2char.json")
    MODEL_CONFIG_PATH: Path = Path("./app/ml_models/model_config.json")
    DEMO_VIDEOS_PATH: Path = Path("../demo_videos")

    # GPU
    USE_GPU: bool = True
    GPU_ID: int = 0

    # Database
    DATABASE_URL: str = "sqlite:///./lipreading.db"
    SQLALCHEMY_ECHO: bool = False

    # Redis/Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # Inference
    MAX_FRAMES: int = 75
    FRAME_HEIGHT: int = 50
    FRAME_WIDTH: int = 100
    FPS_TARGET: int = 25
    BLANK_IDX: int = 0
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()