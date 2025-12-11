"""
Demo Video Test Script
======================
Tests the lip reading API with demo videos.
Run this after starting the backend server.

Usage:
    python test_demo.py
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("\n[1] Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/api/health/", timeout=10)
        data = response.json()
        print(f"    Status: {data.get('status', 'unknown')}")
        print(f"    Model Loaded: {data.get('model_loaded', False)}")
        print(f"    GPU Available: {data.get('gpu_available', False)}")
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def list_demo_videos():
    """List available demo videos"""
    print("\n[2] Listing Demo Videos...")
    try:
        response = requests.get(f"{BASE_URL}/api/videos/demo", timeout=10)
        data = response.json()
        videos = data.get('demo_videos', [])
        print(f"    Found {len(videos)} demo videos:")
        for v in videos:
            print(f"      - {v}")
        return videos
    except Exception as e:
        print(f"    ERROR: {e}")
        return []

def test_demo_video(video_name: str):
    """Test inference on a demo video"""
    print(f"\n[3] Testing Inference on: {video_name}")
    try:
        response = requests.post(f"{BASE_URL}/api/videos/demo/{video_name}", timeout=120)
        
        if response.status_code != 200:
            print(f"    ERROR: {response.status_code} - {response.text}")
            return False
        
        data = response.json()
        print(f"    ✓ Predicted Text: '{data.get('predicted_text', 'N/A')}'")
        print(f"    ✓ Confidence: {data.get('confidence_score', 0) * 100:.1f}%")
        print(f"    ✓ Frames: {data.get('frames_detected', 0)}")
        print(f"    ✓ Processing Time: {data.get('processing_time', 0):.2f}s")
        return True
    except requests.exceptions.Timeout:
        print("    ERROR: Request timed out (inference may take a while)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

def main():
    print("=" * 50)
    print("Lip Reading API - Demo Test")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ API is not healthy. Please start the backend server first.")
        print("   Run: cd backend && python -m uvicorn app.main:app --port 8000")
        sys.exit(1)
    
    # List demo videos
    videos = list_demo_videos()
    if not videos:
        print("\n❌ No demo videos found.")
        sys.exit(1)
    
    # Test first demo video
    video_to_test = videos[0]
    if test_demo_video(video_to_test):
        print("\n" + "=" * 50)
        print("✓ Demo test completed successfully!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Demo test failed.")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()
