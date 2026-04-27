# Optical Music Recognition (OMR) using CRNN + CTC

**Group Members:** [Add your group member names here]

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

This project requires Python 3.10 specifically. TensorFlow 2.10.1 does not install or run correctly on Python 3.11 or above. Before running the commands below, ensure Python 3.10 is installed on your machine. You can download it from https://www.python.org/downloads/release/python-3100/.

Open the VS Code Terminal (Ctrl + ~) and run these commands in order:

```bash
# 1. Create a virtual environment using Python 3.10 explicitly
# On Windows (adjust path if your Python 3.10 install location differs):
py -3.10 -m venv .venv

# On Mac/Linux:
python3.10 -m venv .venv

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

> Note on TensorFlow: Version 2.10.1 pinned to Python 3.10 is required to maintain compatibility with the pre-trained `.meta` weight files and the `tf.compat.v1` behavior used in this project. Newer versions of TensorFlow (2.15+) and newer versions of Python (3.11+) have both removed compatibility layers needed for these weight files.

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

This project uses the PrIMuS (Printed Images of Music Staves) dataset, publicly available from the University of Alicante:

- Official page: https://grfia.dlsi.ua.es/primus/

Due to the size of the full dataset, 860 sample images have been included directly in this repository inside the `Corpus/` folder as a representative subset. This is sufficient to run the pipeline and evaluate the model. If you wish to train or evaluate on the full corpus, download the complete dataset from the link above and extract it into the `Corpus/` folder, replacing the sample contents.

---

## Running the Project

1. Select your interpreter: Press Ctrl + Shift + P, type "Python: Select Interpreter", and choose the one labeled `.venv` with Python 3.10.

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
- Tensors not found: Confirm you are using TensorFlow 2.10.x on Python 3.10 exactly. Versions 2.15 and above, or Python 3.11 and above, have removed the `tf.compat.v1` layers required to load the pre-trained weight files.
- `KeyError` on vocabulary: Ensure `vocabulary_agnostic.txt` is the complete file from the PrIMuS authors. Any missing symbols will cause a key lookup failure during encoding.
- Wrong interpreter selected: If `import tensorflow` fails despite installation, press Ctrl + Shift + P, select "Python: Select Interpreter", and confirm the `.venv` environment is active and shows Python 3.10.
- Wrong Python version in venv: If you accidentally created the venv with the wrong Python version, delete the `.venv` folder and re-run the creation command using `py -3.10` or `python3.10` explicitly.
