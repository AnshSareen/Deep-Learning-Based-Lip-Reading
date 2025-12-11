@echo off
title Lip Reading Backend Auto Setup (Fixed)
echo ========================================
echo   LIP READING BACKEND AUTO SETUP TOOL
echo ========================================

:: Force script to run in backend folder
cd /d %~dp0

:: Step 1 - Check Python 3.10
echo.
echo [1/7] Checking for Python 3.10...
py -3.10 --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python 3.10 NOT found. Installing via winget...
    winget install --id Python.Python.3.10 -e
    echo.
    echo PLEASE RESTART THIS SCRIPT AFTER INSTALLATION COMPLETES!
    pause
    exit
)
echo Python 3.10 Found.

:: Step 2 - Create Virtual Environment
echo.
echo [2/7] Creating virtual environment...
py -3.10 -m venv venv

:: Step 3 - Activate venv
echo.
echo [3/7] Activating virtual environment...
call venv\Scripts\activate

:: Step 4 - Upgrade pip (FIXED)
echo.
echo [4/7] Upgrading pip properly...
python -m pip install --upgrade pip

:: Step 5 - Fix mediapipe version WITHOUT PowerShell (FIXED)
echo.
echo [5/7] Fixing mediapipe version in requirements.txt...
findstr /v "mediapipe" requirements.txt > temp.txt
echo mediapipe==0.10.21>> temp.txt
move /y temp.txt requirements.txt

:: Step 6 - Install all dependencies
echo.
echo [6/7] Installing dependencies...
pip install -r requirements.txt

chcp 65001

:: Step 7 - Run Backend Server
echo.
echo [7/7] Starting FastAPI Server...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
