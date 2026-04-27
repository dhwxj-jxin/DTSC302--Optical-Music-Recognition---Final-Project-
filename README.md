# Optical Music Recognition (OMR) using CRNN + CTC

**Group Members:** Dhwaj Jain & Prakriti

---

## Project Overview

This project implements an Optical Music Recognition (OMR) system that transcribes scanned sheet music images into machine-readable music notation. The pipeline uses classical image processing (OpenCV) for preprocessing and a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss for sequence recognition. The system is trained and evaluated on the PrIMuS dataset using agnostic encoding.

---

## Setup Instructions

🛠 VS Code Setup (Standard Python)
1. Recommended Extensions
Python (by Microsoft)
Jupyter (if using notebooks)
2. Environment Creation & Installation
Open your VS Code Terminal (Ctrl + ~) and run these commands in order:
code
Bash
# 1. Create a virtual environment named '.venv'
python -m venv .venv

# 2. Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install required libraries
pip install tensorflow==2.10.1 opencv-python numpy matplotlib pandas scikit-learn
Note on TensorFlow: Version 2.10.1 is required to maintain compatibility with the pre-trained .meta weight files and the tf.compat.v1 behavior used in this project.
Project File Structure
VS Code relies on correct relative paths. Ensure your workspace looks exactly like this:
code
Text
OMR_Project/
├── .venv/                           # Created by the commands above
├── main.py                          # Your primary Python script
├── vocabulary_agnostic.txt          # REQUIRED: The official 758-symbol list
├── agnostic_model.meta              # Pre-trained Weights (Part 1)
├── agnostic_model.index             # Pre-trained Weights (Part 2)
├── agnostic_model.data-00000-of-00001 # Pre-trained Weights (Part 3)
├── Corpus/                          # Folder containing PrIMuS data
│   ├── 000051652-1_2_1/
│   │   ├── 000051652-1_2_1.png
│   │   └── 000051652-1_2_1.agnostic
│   └── ... (other folders)
└── README.md
Important:
Root Folder: Open the OMR_Project folder directly in VS Code (File > Open Folder).
Weights: The three agnostic_model files must be in the same folder as main.py.
Running the Project
Select your interpreter: Press Ctrl + Shift + P, type "Python: Select Interpreter", and choose the one labeled .venv.
Run the script:
code
Bash
python main.py
Pipeline Overview
Data Loading: Loads the PrIMuS corpus (Images + Agnostic labels).
Model Restoration: Restores the TensorFlow v1 graph and pre-trained weights.
Preprocessing: Resizes images to 128px height and normalizes pixels.
Inference: The CRNN processes the image horizontally, and CTC decoding produces the note sequence.
Evaluation: Calculates SER (Symbol Error Rate), CER (Character Error Rate), and LER (Label Error Rate).
Ablation Study: Visualizes the importance of Bi-LSTMs and Data Augmentation.
What is Left / To-Do
To complete or further optimize this project, the following steps are remaining:

MIDI Integration: Link the output strings to a MIDI synthesis library to play the recognized music.

Sliding Window Inference: Improve accuracy on extremely long staves by processing them in chunks.

Robust Preprocessing: Add logic to handle "noisy" scans (skew correction, deskewing) for real-world photos.

GUI: Build a simple VS Code-based interface or a web-app using Streamlit for easy image uploading.
Troubleshooting
MemoryError: If your computer has less than 8GB RAM, find the sample_limit in the code and set it to 100.
Tensors not found: Ensure you are using TensorFlow 2.10.x. Newer versions (2.15+) have removed certain compatibility layers needed for the 2018 weight files.
KeyError: Ensure vocabulary_agnostic.txt is the full file provided by the PrIMuS authors.
