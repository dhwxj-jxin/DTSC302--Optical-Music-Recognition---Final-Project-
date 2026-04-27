# Optical Music Recognition (OMR) using CRNN + CTC

**Group Members:** Dhwaj Jain & Prakriti

---

## Project Overview

This project implements an Optical Music Recognition (OMR) system that transcribes scanned sheet music images into machine-readable music notation. The pipeline uses classical image processing (OpenCV) for preprocessing and a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss for sequence recognition. The system is trained and evaluated on the PrIMuS dataset using agnostic encoding.

---

## VS Code Setup (Standard Python)

### Recommended Extensions

Install the following extensions from the VS Code Extensions panel (Ctrl + Shift + X):

- Python (by Microsoft)
- Jupyter (if using notebooks)

### Environment Creation and Installation

Open the VS Code Terminal (Ctrl + ~) and run these commands in order:

```bash
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
```

> Note on TensorFlow: Version 2.10.1 is required to maintain compatibility with the pre-trained `.meta` weight files and the `tf.compat.v1` behavior used in this project. Newer versions (2.15+) have removed compatibility layers needed for these 2018 weight files.

---

## Project File Structure

VS Code relies on correct relative paths. Ensure your workspace looks exactly like this:

```
OMR_Project/
├── .venv/                              # Created by the commands above
├── main.py                             # Your primary Python script
├── vocabulary_agnostic.txt             # REQUIRED: The official 758-symbol list
├── agnostic_model.meta                 # Pre-trained weights (Part 1)
├── agnostic_model.index                # Pre-trained weights (Part 2)
├── agnostic_model.data-00000-of-00001  # Pre-trained weights (Part 3)
├── Corpus/                             # Folder containing PrIMuS data
│   ├── 000051652-1_2_1/
│   │   ├── 000051652-1_2_1.png
│   │   └── 000051652-1_2_1.agnostic
│   └── ... (other sample folders)
└── README.md
```

Important notes:

- Root folder: Open the `OMR_Project` folder directly in VS Code via File > Open Folder.
- Weights: The three `agnostic_model` files must be in the same folder as `main.py`.
- Vocabulary: `vocabulary_agnostic.txt` must be the full file provided by the PrIMuS authors. A truncated or custom file will cause a `KeyError` at runtime.

---

## Data Access

This project uses the PrIMuS (Printed Images of Music Staves) dataset, publicly available from the University of Alicante.

- Official page: https://grfia.dlsi.ua.es/primus/

After downloading, extract the corpus into the `Corpus/` folder as shown in the file structure above. Each subfolder corresponds to one music staff sample and must contain both a `.png` image and a `.agnostic` label file.

---

## Running the Project

1. Select your interpreter: Press Ctrl + Shift + P, type "Python: Select Interpreter", and choose the one labeled `.venv`.

2. Run the script from the terminal:

```bash
python main.py
```

---

## Pipeline Overview

The script executes the following stages in order:

1. Data loading: Loads the PrIMuS corpus (images and agnostic labels) from the `Corpus/` folder.
2. Model restoration: Restores the TensorFlow v1 graph and pre-trained weights from the `agnostic_model` files.
3. Preprocessing: Binarizes images using Otsu thresholding, removes staff lines via morphological operations, crops melody strips, resizes to 128px height, and normalizes pixel values to [0, 1].
4. Inference: The CRNN processes the image horizontally and CTC greedy decoding produces the note token sequence.
5. Evaluation: Calculates SER (Symbol Error Rate), CER (Character Error Rate), and LER (Label Error Rate) over the validation subset.
6. Visualization: Displays CNN feature maps, preprocessing stages, and per-sample prediction comparisons.

---

## What Is Left / To-Do

The following steps remain to complete or further optimize this project:

- MIDI integration: Link the output token strings to a MIDI synthesis library to play the recognized music.
- Sliding window inference: Improve accuracy on extremely long staves by processing them in overlapping chunks.
- Robust preprocessing: Add skew correction and deskewing logic to handle real-world photographs of sheet music.
- GUI: Build a simple interface using Streamlit for drag-and-drop image uploading and live transcription output.

---

## Troubleshooting

- `MemoryError`: If your machine has less than 8GB of RAM, locate the `sample_limit` variable in `main.py` and set it to `100`.
- Tensors not found: Confirm you are using TensorFlow 2.10.x exactly. Versions 2.15 and above have removed the `tf.compat.v1` layers required to load the pre-trained weight files.
- `KeyError` on vocabulary: Ensure `vocabulary_agnostic.txt` is the complete file from the PrIMuS authors. Any missing symbols will cause a key lookup failure during encoding.
- Wrong interpreter selected: If `import tensorflow` fails despite installation, press Ctrl + Shift + P, select "Python: Select Interpreter", and confirm the `.venv` environment is active.
