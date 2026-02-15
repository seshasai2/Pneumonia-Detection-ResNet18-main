# Pneumonia-Detection-ResNet18
# Pneumonia Detection — ResNet18 / EfficientNet (transfer learning) + Grad-CAM

**One-line:** Fast, explainable chest X-ray classifier that identifies pneumonia vs normal using transfer learning and Grad-CAM visual explanations.

## Summary
This project trains a lightweight CNN (ResNet18 baseline, optional EfficientNet fine-tune) on the Kaggle *Chest X-Ray Pneumonia* dataset. The pipeline includes data augmentation, mixed-precision fine-tuning, model evaluation (accuracy / recall / precision / F1), and Grad-CAM visualizations for model explainability. A small inference helper and Streamlit/Gradio demo can be used to test images locally.

## Why it matters
In a clinical triage scenario, a fast, explainable model can prioritize likely pneumonia cases for quicker human review — reducing time-to-diagnosis and improving resource allocation. Emphasis is on recall (catching positive cases) with visual heatmaps to increase clinician trust.

## Contents
- `notebook_day13_fast.ipynb` — runnable notebook (baseline + quick fine-tune)
- `fast_outputs/` — saved artifacts: `best_resnet18.pt`, `best_finetuned_resnet18.pt`, confusion matrices, Grad-CAM overlays
- `inference_helper.py` — lightweight script to run predictions on a single image
- `app/streamlit_demo.py` *(optional)* — demo app to upload an X-ray and show prediction + Grad-CAM
- `README.md` — this file

## Dataset
Kaggle: **Chest X-Ray Images (Pneumonia)** — folder structure expected: `train/`, `val/`, `test/`.  
Kaggle link: `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`

## Key steps (what I did)
1. Preprocessing: resize, normalize, augment (RandomResizedCrop, Flip, ColorJitter).  
2. Baseline: ResNet18 with frozen backbone; train classifier head.  
3. Grad-CAM: produce overlay heatmaps for sample predictions to show model focus.  
4. Fine-tune: unfreeze last block + fc, mixed precision training, CosineAnnealingLR scheduler.  
5. Evaluation: Confusion matrix, classification report (precision / recall / F1), multiple saved artifacts.  
6. Deployment: `inference_helper.py` for quick demo; Streamlit app optional.

## How to run (quick)
1. Clone repo and put dataset folder so path `DATA_DIR` points to root with `train/`, `val/`, `test/`.  
2. Create environment:
   ```bash
   pip install -r requirements.txt
   # requirements: torch torchvision matplotlib seaborn scikit-learn opencv-python
Run the quick notebook/script (QUICK mode) to validate:

python fast_day13_14.py   # or open notebook_day13_fast.ipynb and run cells
Run the inference helper:


python fast_outputs/inference_helper.py fast_outputs/best_finetuned_resnet18.pt path/to/xray.jpg


Notes & next steps
For production: train full dataset with IMG_SIZE=224, more epochs, and cross-validation. Export to TorchScript / ONNX for inference speed.

Consider adding class-weighted loss or focal loss if false negatives persist.

For regulatory/clinical use add dataset provenance, patient de-identification checks, and clinical validation steps.

kaggle link:<https://www.kaggle.com/code/seshasai2409/pneumonia-detection-resnet18/notebook?scriptVersionId=272247032>
