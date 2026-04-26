Optical Music Recognition (OMR) using CRNN + CTC
👥 Group Members
Dhwaj Jain
(Add other members here)
📌 Project Overview

This project implements an Optical Music Recognition (OMR) system that converts sheet music images into symbolic representations and MIDI sequences.

The pipeline follows a modern deep learning approach:

Image preprocessing (binarization, staff removal)
Staff segmentation into melody strips
CRNN (CNN + BiLSTM) architecture
CTC loss for sequence prediction
Decoding into musical symbols
Optional MIDI generation
⚙️ Setup Instructions
🔹 1. Clone Repository
git clone <your-repo-link>
cd <repo-name>
🔹 2. Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
🔹 3. Install Dependencies

Create a requirements.txt file with:

numpy
opencv-python
matplotlib
tensorflow
scikit-learn
rapidfuzz

Then run:

pip install -r requirements.txt
📂 Data Access

This project uses the PrIMuS dataset.

🔹 Dataset Structure

Place dataset in:

Corpus/
   ├── 0000xxxx/
   │     ├── image.png
   │     ├── image.agnostic

Each folder must contain:

.png → sheet image
.agnostic → label sequence
🔹 How Data is Loaded

The dataset is loaded using:

data = load_primus_dataset("Corpus")

✔ Automatically:

samples up to 5000 files
loads images + labels
builds training/validation split
🚀 Execution Guide
🔹 Step 1 — Run Main Script
python DTSC302_OpticalMusicRecognition.py
🔹 Step 2 — What Happens Internally
✔ Phase 1 — Data Loading
Loads dataset with progress bar
Splits into train/validation
✔ Phase 2 — Preprocessing
Binarization (Otsu threshold)
Staff line removal
Melody strip extraction
✔ Phase 3 — Tensor Preparation
Resize to 128 × 1024
Normalize pixels
Convert to model-ready tensors
✔ Phase 4 — Model
CNN feature extractor
BiLSTM sequence modeling
CTC loss training
✔ Phase 5 — Inference
Load pretrained model (TF1)
Decode predictions using CTC
Convert tokens → notes
✔ Phase 6 — Evaluation
SER (Symbol Error Rate)
CER (Character Error Rate)
✔ Phase 7 — MIDI Output

Example:

Generated MIDI Sequence: [74, 74, 71, 60, 77, 76, 76]
📊 Output

The system produces:

Predicted musical symbols
Evaluation metrics (SER, CER)
MIDI note sequence
Visualization of processed images
🧠 Key Features
✔ End-to-end CRNN architecture
✔ CTC-based sequence decoding
✔ Staff removal using morphology
✔ Works on real music sheets
✔ Supports pretrained model inference
⚠️ Notes & Limitations
Requires correct dataset structure
Performance depends on training size
TF1 pretrained model used for inference
Accuracy improves with more data + epochs
📌 Future Improvements
Attention-based models (Transformer)
Better pitch detection
Real-time OMR system
Improved augmentation
🎯 Summary

This project demonstrates a complete OMR pipeline, from raw sheet image to musical sequence prediction, combining:

Computer Vision + Deep Learning + Sequence Modeling
