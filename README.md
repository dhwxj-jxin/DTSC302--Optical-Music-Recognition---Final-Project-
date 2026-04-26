# Optical Music Recognition (OMR) using CRNN + CTC

**Group Members:** [Add your group member names here]

---

## Project Overview

This project implements an Optical Music Recognition (OMR) system that transcribes scanned sheet music images into machine-readable music notation. The pipeline uses classical image processing (OpenCV) for preprocessing and a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss for sequence recognition. The system is trained and evaluated on the PrIMuS dataset using agnostic encoding.

---

## Setup Instructions

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine
- Python 3.10 (managed via the conda environment below)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### Step 2: Create and Activate the Conda Environment

An `environment.yml` file is provided to reproduce the exact environment.

```bash
conda env create -f environment.yml
conda activate omr_env
```

This installs:
- Python 3.10
- TensorFlow 2.10.1 (with Keras 2.10.0, required for `.meta`/`.data` weight files via `tf.compat.v1`)
- OpenCV
- NumPy
- Matplotlib

> Note: TensorFlow 2.10 is the last version with native Windows GPU support. If running on Linux/macOS, the same version works on CPU. For GPU acceleration, ensure CUDA 11.2 and cuDNN 8.1 are installed separately.

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected output: `2.10.1`

---

## Data Access

This project uses the **PrIMuS (Printed Images of Music Staves)** dataset.

### Download

The dataset is publicly available from the University of Alicante:

- Official page: https://grfia.dlsi.ua.es/primus/
- Direct download: Request access or download the corpus archive from the above page.

### Directory Structure

After downloading, extract the corpus so the folder structure looks like this:

```
project-root/
    Corpus/
        000051652-1_2_1/
            000051652-1_2_1.png
            000051652-1_2_1.agnostic
        000051652-1_2_2/
            ...
    agnostic_model.meta
    agnostic_model.data-00000-of-00001
    agnostic_model.index
    vocabulary_agnostic.txt
    main.py
    environment.yml
    README.md
```

Each subfolder inside `Corpus/` corresponds to one music staff sample and contains:
- A `.png` image of the staff
- A `.agnostic` file with the ground-truth symbol sequence

### Pre-trained Model Weights

The pre-trained CRNN weights (`agnostic_model.meta`, `.data`, `.index`) and the vocabulary file (`vocabulary_agnostic.txt`) must be placed in the project root directory. These are loaded at inference time using TensorFlow 1.x's `tf.train.Saver`.

If the weights are not included in the repository due to size constraints, contact the project authors or download them from the same PrIMuS resource page.

---

## Execution Guide

All code is contained in a single script: `main.py`

### Run the Full Pipeline

```bash
conda activate omr_env
python main.py
```

### What the Script Does (in order)

1. **Data Loading** - Loads up to 5,000 random samples from the `Corpus/` directory.
2. **Vocabulary Building** - Scans training labels to build a token-to-index mapping and saves it to `vocab.json`.
3. **Preprocessing** - For each sample: binarizes the image (Otsu thresholding), removes staff lines via morphological operations, and crops individual melody strips.
4. **Tensor Preparation** - Resizes each strip to a fixed `128 x 1024` canvas and normalizes pixel values to `[0, 1]`.
5. **CRNN Model Definition** - Builds a CRNN architecture (4 CNN blocks + 2 Bidirectional LSTM layers) compiled with CTC loss.
6. **Pre-trained Weight Loading** - Restores the pre-trained agnostic model from `.meta`/`.data` files using TensorFlow 1.x compatibility mode.
7. **Inference** - Runs CTC greedy decoding on a test image and prints the predicted music token sequence.
8. **Evaluation** - Computes Symbol Error Rate (SER) and Character Error Rate (CER) over a 50-sample validation subset.
9. **Per-sample Prediction** - Picks a random validation sample, prints ground truth vs. prediction, and shows the image.
10. **MIDI Export** - Maps predicted tokens to a basic MIDI pitch sequence (printed to console).

### Modifying Key Parameters

| Parameter | Location in `main.py` | Default |
|---|---|---|
| Dataset sample limit | `load_primus_dataset()` call | 5000 |
| Train/val split | `split_idx` | 80% / 20% |
| Target image size | `TARGET_HEIGHT`, `TARGET_WIDTH` | 128, 1024 |
| Max label length | `MAX_LABEL_LEN` | 100 |
| Evaluation subset size | `test_subset = val_data[:50]` | 50 |
| Test image path | `TEST_IMAGE_PATH` | `Corpus/000051652-1_2_1/...` |

---

## Output Files

- `vocab.json` - Token-to-index vocabulary generated from the training set.
- Console output includes SER, CER metrics, and per-sample predictions.
- Matplotlib windows display preprocessing stages, feature maps, and transcription results.

---

## Notes

- The script uses `tensorflow.compat.v1` with `tf.disable_v2_behavior()` to load the pre-trained `.meta` weights. This is intentional and required for compatibility with the PrIMuS model checkpoint format.
- The CRNN model defined in the script (`build_crnn_model`) is used for architecture reference and CNN feature visualization. Inference is performed using the restored pre-trained graph.
- Staff line removal and strip cropping are applied during preprocessing but the pre-trained model's inference path (`predict_music`) uses the full resized image directly, consistent with the original PrIMuS evaluation protocol.
