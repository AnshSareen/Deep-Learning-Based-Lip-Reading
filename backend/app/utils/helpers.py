import os
import uuid
from pathlib import Path
from datetime import datetime

def generate_unique_filename(original_filename: str, upload_type: str = "video") -> str:
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    ext = Path(original_filename).suffix
    return f"{upload_type}_{timestamp}_{unique_id}{ext}"

def ensure_directory_exists(path: Path) -> None:
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return Path(filename).suffix.lstrip('.').lower()

def validate_file_extension(filename: str, allowed_extensions) -> bool:
    """Validate file extension"""
    ext = get_file_extension(filename)
    return ext in allowed_extensions

def format_timestamp(dt: datetime) -> str:
    """Format timestamp for API response"""
    return dt.isoformat()

def clean_old_files(directory: Path, hours: int = 24) -> int:
    """Clean files older than specified hours"""
    from datetime import timedelta
    import time
    
    if not directory.exists():
        return 0
    
    cutoff_time = time.time() - (hours * 3600)
    deleted_count = 0
    
    for file in directory.glob('*'):
        if file.is_file() and os.path.getctime(file) < cutoff_time:
            try:
                file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    
    return deleted_count