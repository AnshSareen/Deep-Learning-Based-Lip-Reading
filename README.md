# 👄 Deep Learning Based Lip Reading System

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18+-61DAFB.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

A complete end-to-end system for **Visual Speech Recognition** (Lip Reading). This application uses a deep learning model (3D CNN + 2D CNN + Bi-LSTM) to predict speech text solely from video of lip movements. It features a modern React frontend, a FastAPI backend, and real-time inference capabilities.

---

## ✨ Features

- **🎥 Video Inference**: Upload your own video files (MP4, AVI, MOV) for instant lip reading.
- **⚡ Real-time Processing**: Fast inference using an optimized TorchScript model.
- **📊 Performance Metrics**: Automatically calculates **WER** (Word Error Rate) and **CER** (Character Error Rate) when ground truth is available.
- **🖥️ Modern Web UI**: Responsive interface built with React and Vite for easy interaction.
- **🧪 Demo Mode**: Includes pre-loaded sample videos from the GRID dataset to test immediately.
- **🔌 REST API**: Full-featured API for integrating lip reading into other applications.

---

## 🏗️ Architecture

The system consists of three main components:

- **Frontend**: A React application that handles video uploads and displays results/metrics.
- **Backend**: A FastAPI server that processes videos, runs the PyTorch model, and returns predictions.
- **AI Model**: A hybrid architecture:
  - **Spatiotemporal Feature Extraction**: 3D CNN frontend
  - **Spatial Feature Extraction**: 2D CNN backend
  - **Sequence Modeling**: 2-layer Bidirectional LSTM
  - **Decoding**: CTC (Connectionist Temporal Classification) Loss

---

## 🛠️ Project Structure

```bash
lipreading_model_deployment/
├── backend/                 # Python FastAPI backend
│   ├── app/                 # Application logic & API routers
│   ├── uploads/             # Temp storage for uploaded videos
│   └── requirements.txt     # Python dependencies
├── frontend/                # React frontend application
│   ├── src/                 # UI components and logic
│   └── package.json         # Node.js dependencies
├── demo_videos/             # Sample videos for testing
├── transcripts/             # Ground truth texts for GRID dataset
├── model_deploy.torchscript # Trained PyTorch model
├── inference_full.py        # 🖥️ CLI Inference Script
├── start.bat                # 🚀 Windows One-Click Launch Script
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))

### Option 1: Quick Start (Windows) ~ Recommended

We have provided a smart launch script that handles everything for you.

1. Double-click the **`start.bat`** file in the root directory.
2. The script will:
   - Verify your environment.
   - Install all Python and Node.js dependencies (if missing).
   - Launch both the Backend (Port 8000) and Frontend (Port 5173).
   - Automatically open your browser to the application.

### Option 2: Manual Installation

If you are on Linux/Mac or prefer manual setup:

#### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
# Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup
Open a new terminal:
```bash
cd frontend
npm install
# Run the development server
npm run dev
```

Visit **http://localhost:5173** to use the app.

---

## � Command Line Interface (CLI)

You can run predictions directly from your terminal without starting the web server.

**Run on a video file:**
```bash
python inference_full.py --video path/to/video.mp4
```

**Run with manual transcript (for metric calculation):**
```bash
python inference_full.py --video path/to/video.mp4 --transcript "place red at p six please"
```

The script will output the predicted text, and if a ground truth is provided (or found in `transcripts/`), it will calculate and show the WER and CER.

---

## 📖 Web Application Usage

1. **Upload Video**: Click standard upload area to select a video file of a person speaking. The video should focus on the face/lips.
2. **View Results**: The model will output:
   - **Transcript**: The predicted text.
   - **Metrics**: If you are using a demo video or provide ground truth, it will show WER and CER percentages.
3. **Try Demos**: Use the "Try Demo Videos" button to load sample videos from the GRID Corpus to see the model in action immediately.

---

## 📊 Model Performance

The model was trained on the **GRID Corpus** (34,000 videos).

| Metric | Score | Description |
|:-------|:-----:|:------------|
| **CER (Character Error Rate)** | **~5.6%** | Percentage of incorrect characters. |
| **WER (Word Error Rate)** | **~12.4%** | Percentage of incorrect words. |

*Note: Performance is best on videos with clear lighting, frontal view, and similar framing to the GRID dataset.*

---

## 🔗 API Reference

The backend exposes the following REST endpoints:

- `GET /api/health`: Check system status.
- `POST /api/videos/upload`: Upload a file for processing.
- `GET /api/videos/demo`: Get list of available demo videos.
- `POST /api/videos/demo/{video_name}`: Run inference on a specific demo video.

Full interactive documentation is available at **http://localhost:8000/api/docs** when the server is running.

---

## 📄 License

This project is open source and available for educational and research purposes.
