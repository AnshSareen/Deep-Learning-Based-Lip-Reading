# Lip Reading Project Package

This package contains the complete lip reading system, including the trained model, inference scripts, and demo videos.

## 📂 Contents

*   **`model_deploy.torchscript`**: The trained deep learning model (30 epochs, 34k videos).
*   **`inference_full.py`**: Script to run the model on new videos.
*   **`train_full.py`**: The original training script used to create the model.
*   **`preprocess_grid.py`**: The script used to process the raw GRID dataset.
*   **`idx2char.json`**: Mapping file for decoding model outputs.
*   **`demo_videos/`**: 5 sample videos from the GRID dataset for testing.

## 🚀 How to Run Inference

1.  **Install Dependencies:**
    You need Python installed with these libraries:
    ```bash
    pip install torch opencv-python numpy
    ```

2.  **Run on a Demo Video:**
    ```bash
    python inference_full.py --video demo_videos/prap6p.mpg --transcript "place red at p six please"
    ```

3.  **Run on Your Own Video:**
    ```bash
    python inference_full.py --video /path/to/your/video.mp4
    ```

## 🧠 Model Info

*   **Input:** Grayscale video frames of lips (automatically cropped by the script).
*   **Output:** Text transcription.
*   **Performance:** ~5.6% Character Error Rate (CER) on test set.
*   done

## 🛠️ Training (Optional)

If you want to retrain the model:
1.  Download the GRID dataset.
2.  Run `preprocess_grid.py` to extract frames.
3.  Run `train_full.py` to train the model.
