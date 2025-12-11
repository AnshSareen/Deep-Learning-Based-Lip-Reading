"""
Setup script to copy model files to the correct location.
Run this once to set up the ml_models directory.
"""
import shutil
import os
from pathlib import Path

# Define paths
ROOT = Path(__file__).parent.parent
ML_MODELS_DIR = Path(__file__).parent / "app" / "ml_models"
MODEL_SRC = ROOT / "model_deploy.torchscript"
IDX2CHAR_SRC = ROOT / "idx2char.json"

def setup_ml_models():
    print("=" * 50)
    print("Setting up ML Models Directory")
    print("=" * 50)
    
    # Create ml_models directory
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {ML_MODELS_DIR}")
    
    # Copy model
    if MODEL_SRC.exists():
        dest = ML_MODELS_DIR / "model_deploy.torchscript"
        if MODEL_SRC.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(MODEL_SRC, dest)
        else:
            shutil.copy2(MODEL_SRC, dest)
        print(f"✓ Copied model to: {dest}")
    else:
        print(f"✗ Model not found at: {MODEL_SRC}")
    
    # Copy idx2char.json
    if IDX2CHAR_SRC.exists():
        dest = ML_MODELS_DIR / "idx2char.json"
        shutil.copy2(IDX2CHAR_SRC, dest)
        print(f"✓ Copied idx2char.json to: {dest}")
    else:
        print(f"✗ idx2char.json not found at: {IDX2CHAR_SRC}")
    
    # Verify
    print("\n" + "=" * 50)
    print("Verification:")
    for item in ML_MODELS_DIR.iterdir():
        if item.is_dir():
            print(f"  [DIR]  {item.name}")
        else:
            print(f"  [FILE] {item.name} ({item.stat().st_size} bytes)")
    print("=" * 50)

if __name__ == "__main__":
    setup_ml_models()
