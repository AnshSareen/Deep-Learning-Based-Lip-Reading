@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Lip Reading Model - Launch Script

echo.
echo ========================================================
echo      LIP READING MODEL DEPLOYMENT - SMART LAUNCHER
echo ========================================================
echo.

:: ============================================
:: STEP 1: Check if running from correct directory
:: ============================================
if not exist "backend\app\main.py" (
    echo [ERROR] Please run this script from the project root directory!
    echo         Expected to find: backend\app\main.py
    echo.
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] Frontend not found!
    echo         Expected to find: frontend\package.json
    echo.
    pause
    exit /b 1
)

echo [OK] Project structure verified
echo.

:: ============================================
:: STEP 2: Check Python installation
:: ============================================
echo [CHECKING] Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo         Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

:: ============================================
:: STEP 3: Check Node.js installation
:: ============================================
echo [CHECKING] Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH!
    echo         Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

for /f "tokens=1" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
echo [OK] Node.js %NODE_VERSION% found
echo.

:: ============================================
:: STEP 4: Check and install backend dependencies
:: ============================================
echo [CHECKING] Backend dependencies...
cd backend

:: Check if key packages are installed
python -c "import fastapi; import uvicorn; import pydantic_settings; import torch" >nul 2>&1
if errorlevel 1 (
    echo [INSTALLING] Backend dependencies from requirements.txt...
    echo              This may take a few minutes on first run...
    pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies!
        cd ..
        pause
        exit /b 1
    )
    echo [OK] Backend dependencies installed
) else (
    echo [OK] Backend dependencies already installed
)

cd ..
echo.

:: ============================================
:: STEP 5: Check and install frontend dependencies  
:: ============================================
echo [CHECKING] Frontend dependencies...
cd frontend

if not exist "node_modules" (
    echo [INSTALLING] Frontend dependencies...
    echo              Running npm install...
    npm install -q
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies!
        cd ..
        pause
        exit /b 1
    )
    echo [OK] Frontend dependencies installed
) else (
    echo [OK] Frontend dependencies already installed
)

cd ..
echo.

:: ============================================
:: STEP 6: Check model files
:: ============================================
echo [CHECKING] Model files...
if not exist "idx2char.json" (
    echo [WARNING] idx2char.json not found - model may not work correctly
) else (
    echo [OK] Character mapping found
)

if not exist "backend\app\ml_models\model_deploy.torchscript.zip" (
    if exist "model_deploy.torchscript.zip" (
        echo [OK] Model file found in root directory
    ) else if exist "model_deploy.torchscript" (
        echo [OK] Model directory found
    ) else (
        echo [WARNING] Model file not found - predictions will fail
    )
) else (
    echo [OK] Model file found
)
echo.

:: ============================================
:: STEP 7: Start Backend Server
:: ============================================
echo ========================================================
echo                   STARTING SERVERS
echo ========================================================
echo.

echo [1/2] Starting Backend Server on port 8000...
start "Lip Reading - Backend" cmd /k "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

:: Wait for backend to initialize
echo      Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

:: ============================================
:: STEP 8: Start Frontend Server
:: ============================================
echo [2/2] Starting Frontend Server on port 5173...
start "Lip Reading - Frontend" cmd /k "cd frontend && npm run dev"

:: Wait for frontend to start
timeout /t 3 /nobreak >nul

:: ============================================
:: STEP 9: Display endpoints and open browser
:: ============================================
echo.
echo ========================================================
echo                  SERVERS ARE RUNNING!
echo ========================================================
echo.
echo   FRONTEND APPLICATION:
echo   ---------------------
echo   URL:  http://localhost:5173
echo.
echo   BACKEND API:
echo   ---------------------
echo   API Base:     http://localhost:8000
echo   API Docs:     http://localhost:8000/api/docs
echo   Health Check: http://localhost:8000/api/health
echo.
echo   DEMO VIDEO ENDPOINTS:
echo   ---------------------
echo   List demos:   GET  http://localhost:8000/api/videos/demo
echo   Run demo:     POST http://localhost:8000/api/videos/demo/{video_name}
echo.
echo ========================================================
echo   Press any key to open the frontend in your browser...
echo   Close the server windows to stop the application.
echo ========================================================
echo.

pause

:: Open frontend in default browser
start http://localhost:5173

echo.
echo Script completed. Servers are running in separate windows.
echo.
