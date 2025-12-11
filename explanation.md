# Lip Reading Model - Project Explanation

## What is this project?

This is a **Lip Reading AI Model** that can transcribe speech by analyzing lip movements in videos. The system uses deep learning to predict what someone is saying by watching their lips move.

---

## Project Structure

```
lipreading_model_deployment/
├── backend/                 # Python FastAPI server
│   ├── app/                # API code
│   │   ├── services/       # Inference engine, video processing
│   │   ├── routers/        # API endpoints
│   │   └── ml_models/      # TorchScript model files
│   └── requirements.txt    # Python dependencies
├── frontend/               # React web interface
│   └── src/               # React components
├── demo_videos/           # Sample videos for testing
├── transcripts/           # Ground truth transcripts (for WER/CER calculation)
├── idx2char.json          # Character mapping for model output
├── start.bat              # Launch script (Windows)
└── explanation.md         # This file
```

---

## How to Run

### Prerequisites

Before running, make sure you have installed:

1. **Python 3.10 or higher**
   - Download from: https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"
   - Verify: Open Command Prompt and type `python --version`

2. **Node.js 18 or higher**
   - Download from: https://nodejs.org/
   - Choose the LTS version
   - Verify: Open Command Prompt and type `node --version`

---

### Method 1: Quick Start (Recommended)

**Step 1:** Open the project folder in File Explorer

**Step 2:** Double-click `start.bat`

**Step 3:** Wait for the script to:
- Check your Python and Node.js installation
- Install dependencies (only on first run, may take 2-3 minutes)
- Start both servers

**Step 4:** Press any key when prompted to open the browser

**Step 5:** The application opens at http://localhost:5173

---

### Method 2: Manual Start (Command Line)

If you prefer to run manually or the batch file doesn't work:

**Terminal 1 - Backend Server:**
```cmd
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend Server:**
```cmd
cd frontend
npm install
npm run dev
```

**Then open:** http://localhost:5173

---

### Method 3: Run Inference Only (No Web UI)

To run predictions directly from command line without the web interface:

```cmd
python inference_full.py --video demo_videos/bbad3n.mpg
```

This will output the prediction, ground truth, WER, and CER directly in the terminal.

---


## Using the Application

### Option 1: Upload Your Own Video
1. Click the upload area or drag-and-drop a video file
2. Supported formats: MP4, AVI, MOV, MKV, MPG
3. Click "Analyze Video"
4. View the predicted speech text

### Option 2: Try Demo Videos
1. Click "Try Demo Videos"
2. Select one of the pre-loaded demo videos
3. Click "Analyze Video"
4. Results show:
   - **Predicted Text**: What the model thinks was said
   - **Ground Truth**: The actual spoken words (if available)
   - **WER (Word Error Rate)**: % of word-level errors
   - **CER (Character Error Rate)**: % of character-level errors

---

## Understanding the Metrics

| Metric | Meaning | Good Score |
|--------|---------|------------|
| **WER** | Word Error Rate - how many words were wrong | < 20% is good |
| **CER** | Character Error Rate - how many characters were wrong | < 20% is good |

**Color coding:**
- 🟢 Green (0-20%): Excellent
- 🟡 Yellow (20-50%): Moderate  
- 🔴 Red (>50%): Poor

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check if server is running |
| `/api/videos/demo` | GET | List available demo videos |
| `/api/videos/demo/{name}` | POST | Run inference on a demo video |
| `/api/videos/upload` | POST | Upload a new video |
| `/api/docs` | GET | Interactive API documentation |

---

## Requirements

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **~500MB disk space** for dependencies

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python from python.org and add to PATH |
| "Node not found" | Install Node.js from nodejs.org |
| "Model not loaded" | Ensure model files are in `backend/app/ml_models/` |
| Backend fails to start | Run `pip install -r backend/requirements.txt` manually |
| Frontend fails to start | Run `npm install` in the `frontend/` folder |

---

## Technical Details

- **Model**: 3D CNN + BiLSTM with CTC loss (TorchScript)
- **Training**: 30 epochs on 34,000 videos from GRID corpus
- **Input**: 75 frames of grayscale lip region (100x50 pixels)
- **Output**: Predicted text via CTC decoding
- **Backend**: FastAPI (Python)
- **Frontend**: React + Vite
