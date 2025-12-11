# Lip Reading API - Windows Setup Guide

## Prerequisites

- **Python 3.9 or higher** (Download from https://www.python.org/downloads/)
- **CUDA-compatible GPU** (NVIDIA) with CUDA Toolkit installed
- **Git** (optional, for version control)

## Installation Steps

### 1. Extract the Project

Extract the `backend` folder to your desired location, e.g., `C:\Projects\lipreading_backend\`

### 2. Create Virtual Environment

Open **Command Prompt** or **PowerShell** as Administrator and navigate to the backend folder:

```cmd
cd C:\Projects\lipreading_backend
python -m venv venv
```

### 3. Activate Virtual Environment

**Command Prompt:**
```cmd
venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
venv\Scripts\Activate.ps1
```

> **Note:** If you get an execution policy error in PowerShell, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 4. Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** This will install PyTorch with CUDA support. Installation may take 5-10 minutes.

### 5. Configure Environment Variables

Copy the example environment file:
```cmd
copy .env.example .env
```

Edit `.env` file and set:
```env
USE_GPU=True
GPU_ID=0
```

### 6. Place Model Files

Ensure these files are in `app\ml_models\`:
- `model_deploy.torchscript` (38 MB)
- `char2idx.json`
- `idx2char.json`
- `model_config.json`

### 7. Run the Server

```cmd
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/api/docs

## Running as Production

For production (without auto-reload):

```cmd
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Testing the API

Open browser and go to: http://localhost:8000/api/docs

Test the health endpoint:
```cmd
curl http://localhost:8000/api/health/
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true
}
```

## Troubleshooting

### CUDA Not Found
- Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Port Already in Use
Change the port number:
```cmd
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### ModuleNotFoundError
Make sure virtual environment is activated and dependencies are installed.

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── schemas.py
│   ├── ml_models/          # Model files here
│   ├── routers/
│   ├── services/
│   └── utils/
├── uploads/
├── logs/
├── requirements.txt
├── .env
└── SETUP_WINDOWS.md
```
