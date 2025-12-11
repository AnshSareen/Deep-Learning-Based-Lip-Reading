# Lip Reading API Backend

Advanced FastAPI backend for lip reading analysis using deep learning.

## Features

- 🚀 FastAPI with async support
- 🎥 Video processing and frame extraction
- 👄 Lip region detection using MediaPipe
- 🧠 Deep learning inference with PyTorch
- 📊 Confidence scoring
- 🔐 Input validation
- 📝 Comprehensive logging
- 🐳 Docker support
- 🔄 CORS enabled

## Installation

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000