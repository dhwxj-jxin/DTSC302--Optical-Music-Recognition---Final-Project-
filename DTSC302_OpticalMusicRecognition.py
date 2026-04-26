from pathlib import Path
import cv2
import sys
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def load_primus_dataset(corpus_path: str, label_type: str = "agnostic"):

    corpus_path = Path(corpus_path)
    samples = []

    # get total folders first
    all_dirs = [d for d in corpus_path.iterdir() if d.is_dir()]
    sample_size = min(5000, len(all_dirs))
    all_dirs = random.sample(all_dirs, sample_size)
    total = len(all_dirs)

    if total == 0:
        print("No sample folders found")
        return []

    print(f"Total samples: {total}\n")

    for i, sample_dir in enumerate(all_dirs, start=1):

        base = sample_dir.name

        img_path = sample_dir / f"{base}.png"
        label_path = sample_dir / f"{base}.{label_type}"

        if not img_path.exists() or not label_path.exists():
            continue

        # load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # load label
        with open(label_path, "r") as f:
            label = f.read().strip().split()

        samples.append((image, label))

        # ===== PROGRESS BAR =====
        percent = (i / total) * 100
        remaining = total - i

        bar_length = 30
        filled_length = int(bar_length * i // total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        sys.stdout.write(
            f"\rProgress: |{bar}| {percent:6.2f}% "
            f"({i}/{total}) | Remaining: {remaining}"
        )
        sys.stdout.flush()

    print("\n\n✅ Loading complete!")

    return samples
data = load_primus_dataset(r"Corpus") 



# VOCABULARY GENERATOR

def build_vocab(dataset):
    """
    Scans labels to build a unique mapping.
    1. Real symbols are mapped 0 to N-1.
    2. <UNK> is mapped to N.
    3. <BLANK> is mapped to the VERY LAST index (N+1).
    """
    symbol_set = set()
    for _, labels in dataset:
        symbol_set.update(labels)

    sorted_symbols = sorted(list(symbol_set))

    token_to_idx = {}
    idx_to_token = {}

    # 1. Assign real musical symbols starting from index 0
    for i, sym in enumerate(sorted_symbols):
        token_to_idx[sym] = i
        idx_to_token[i] = sym

    # 2. Add <UNK> (Unknown) at the next available index
    unk_idx = len(token_to_idx)
    token_to_idx["<UNK>"] = unk_idx
    idx_to_token[unk_idx] = "<UNK>"

    # 3. Add <BLANK> at the VERY END
    # This is the industry standard fix for the CTC "null label" crash
    blank_idx = len(token_to_idx)
    token_to_idx["<BLANK>"] = blank_idx
    idx_to_token[blank_idx] = "<BLANK>"

    return token_to_idx, idx_to_token, blank_idx

def encode_label(label_seq, token_to_idx):
    """Converts a list of string tokens into a list of integer IDs."""
    unk_id = token_to_idx["<UNK>"]
    return [token_to_idx.get(sym, unk_id) for sym in label_seq]

def save_vocab(token_to_idx, path="vocab.json"):
    with open(path, "w") as f:
        json.dump(token_to_idx, f, indent=2)
    print(f"✅ Vocabulary saved to {path}")


# EXECUTION FLOW: SPLIT & BUILD

# 1. Train/Val Split (80% / 20%)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]

print(f"Data Split Complete:")
print(f"   - Training samples:   {len(train_data)}")
print(f"   - Validation samples: {len(val_data)}")

# 2. Build Vocabulary (Using ONLY training data)
token_to_idx, idx_to_token, BLANK_INDEX = build_vocab(train_data)
NUM_CLASSES = len(token_to_idx)

print(f"\nVocabulary build complete:")
print(f"   - Unique tokens (including UNK/BLANK): {NUM_CLASSES}")
print(f"   - BLANK Index (The Null Label): {BLANK_INDEX}")
print(f"   - Sample tokens: {list(token_to_idx.keys())[0:5]}")

# 3. Save for future inference
save_vocab(token_to_idx)

# 4. Quick Sanity Check on Sample 0
img_check, label_check = train_data[0]
encoded_check = encode_label(label_check, token_to_idx)

print("\n--- Sanity Check ---")
print(f"Original Label: {label_check[:5]}...")
print(f"Encoded IDs:    {encoded_check[:5]}...")

print(f"\nTotal loaded samples: {len(data)}")
img, label = data[4586]
print("\nSample labels:", label[:10])
plt.imshow(img, cmap="gray")
plt.title(" ".join(label[:10]))
plt.axis("off")
plt.show() 


def binarize_image(image):
    """
    Convert grayscale to binary using Otsu. 
    Notes become 255 (white), background becomes 0 (black).
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu automatically calculates the optimal threshold
    _, binary = cv2.threshold(
        image, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary



# Select a random sample from the training set for visualization
sample_idx = random.randint(0, len(train_data) - 1)
raw_img, raw_label = train_data[sample_idx]

# Apply binarization
binary_img = binarize_image(raw_img)

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(2, 1, 1)
plt.title(f"Original Training Sample (Index: {sample_idx})")
plt.imshow(raw_img, cmap="gray")
plt.axis("off")

plt.subplot(2, 1, 2)
plt.title("Binarized (Otsu Inverse) - Notes are now White (255)")
plt.imshow(binary_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# Verify mathematical correctness for the CNN
print(f"Unique pixel values in binary image: {np.unique(binary_img)}")
print(f"Ink pixels (255): {np.sum(binary_img == 255)}")
print(f"Background pixels (0): {np.sum(binary_img == 0)}")


def remove_staff_lines(binary_img):
    """
    Identifies horizontal staff lines and removes them, then repairs broken stems.
    """
    h, w = binary_img.shape

    # Create a long, thin horizontal kernel to detect staff lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))

    # Detect lines only
    detected_lines = cv2.morphologyEx(
        binary_img,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    # Subtract the detected lines from the original binary image
    staff_removed = cv2.subtract(binary_img, detected_lines)

    # Repair step: Staff removal often leaves tiny gaps in note stems.
    # We use a small vertical-leaning ellipse to 'heal' these gaps.
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    staff_removed = cv2.morphologyEx(
        staff_removed,
        cv2.MORPH_CLOSE,
        repair_kernel,
        iterations=1
    )

    return detected_lines, staff_removed

detected_lines, no_staff_img = remove_staff_lines(binary_img)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Step 1: Binary Image (From Phase 2.1)")
plt.imshow(binary_img, cmap="gray")
plt.axis("off")

plt.subplot(3, 1, 2)
plt.title("Step 2: Isolated Staff Lines (Features to be removed)")
plt.imshow(detected_lines, cmap="gray")
plt.axis("off")

plt.subplot(3, 1, 3)
plt.title("Step 3: Staff Removed & Stems Repaired (Final Preprocessed Image)")
plt.imshow(no_staff_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. THE FUNCTIONS (UNCHANGED LOGIC, RECALIBRATED FOR PIPELINE)
# ──────────────────────────────────────────────────────────────────────────────

def detect_staff_rows(binary_img, min_row_height=30, gap_threshold=10):
    """
    Finds horizontal bands that contain musical content (staff rows) 
    using horizontal pixel projection.
    """
    row_sums = np.sum(binary_img, axis=1)
    active = row_sums > 0 # Rows that have any 'ink' (255)

    bands = []
    in_band = False
    y_start = 0
    gap_counter = 0

    for y, is_active in enumerate(active):
        if is_active:
            if not in_band:
                y_start = y
                in_band = True
            gap_counter = 0
        else:
            if in_band:
                gap_counter += 1
                if gap_counter > gap_threshold:
                    y_end = y - gap_counter
                    if (y_end - y_start) >= min_row_height:
                        bands.append((y_start, y_end))
                    in_band = False
                    gap_counter = 0

    if in_band:
        y_end = len(active)
        if (y_end - y_start) >= min_row_height:
            bands.append((y_start, y_end))

    return bands

def crop_melody_strips(binary_img, padding=5):
    """
    Crops each detected staff row into a tight horizontal strip.
    """
    h, w = binary_img.shape
    bands = detect_staff_rows(binary_img)

    strips = []
    offsets = []

    for (y0, y1) in bands:
        # Add padding but stay within image boundaries
        y0_pad = max(0, y0 - padding)
        y1_pad = min(h, y1 + padding)

        strip = binary_img[y0_pad:y1_pad, :] # Extract the full-width strip
        strips.append(strip)
        offsets.append((0, y0_pad))

    return strips, offsets

# ──────────────────────────────────────────────────────────────────────────────
# 2. APPLICATION ON THE PREVIOUSLY PROCESSED IMAGE
# ──────────────────────────────────────────────────────────────────────────────

# We use 'no_staff_img' created in Phase 2.2
strips, offsets = crop_melody_strips(no_staff_img)

print(f"Step 2.3 Results:")
print(f"   - Staff rows detected: {len(strips)}")

for i, (strip, (x0, y0)) in enumerate(zip(strips, offsets)):
    print(f"   - Strip {i}: shape={strip.shape} | global offset=(y:{y0})")

# ──────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(strips), 1, figsize=(15, 3 * len(strips)), squeeze=False)

for i, (strip, (x0, y0)) in enumerate(zip(strips, offsets)):
    axes[i, 0].imshow(strip, cmap="gray")
    axes[i, 0].set_title(f"Final Melody Strip {i} (Ready for Phase 3: Tensor Prep)")
    axes[i, 0].axis("off")

plt.tight_layout()
plt.show() 


# ============================================================
# PHASE 3: TENSOR PREPARATION (Based on Paper Page 9/10)
# ============================================================

# The paper uses 128px height. 
# We'll use 1024 as a standard wide width to accommodate most PrIMuS staves.
TARGET_HEIGHT = 128 
TARGET_WIDTH = 1024 

def prepare_tensor_for_model(strip_img):
    """
    Implements the rescaling logic from Section 'Implementation Details' (Page 9).
    1. Rescale to 128px height.
    2. Maintain aspect ratio.
    3. Normalize pixels to [0, 1].
    """
    h, w = strip_img.shape
    
    # 1. Calculate new width maintaining aspect ratio
    aspect_ratio = w / h
    new_w = int(TARGET_HEIGHT * aspect_ratio)
    
    # 2. Resize (using INTER_AREA for high quality downsampling)
    resized = cv2.resize(strip_img, (min(new_w, TARGET_WIDTH), TARGET_HEIGHT), 
                         interpolation=cv2.INTER_AREA)
    
    # 3. Create a blank canvas (padding) to ensure consistent tensor shape
    # The paper uses variable width, but for batch training in Keras/TF, 
    # we pad the remaining width with 0 (black background).
    canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    canvas[:, :resized.shape[1]] = resized / 255.0  # Normalize to [0, 1]
    
    return canvas.reshape(TARGET_HEIGHT, TARGET_WIDTH, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Visualizing the Next Step
# ──────────────────────────────────────────────────────────────────────────────

# Take the first strip from your previous step (Phase 2.3)
demo_strip = strips[0]
model_input_tensor = prepare_tensor_for_model(demo_strip)

print(f"Original Strip Shape: {demo_strip.shape}")
print(f"Tensor Shape for CRNN: {model_input_tensor.shape}")

plt.figure(figsize=(15, 2))
plt.imshow(model_input_tensor.squeeze(), cmap='gray')
plt.title("Phase 3 Result: Rescaled & Normalized Tensor (Ready for CRNN)")
plt.axis('off')
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np

# ============================================================
# PHASE 4.1: BUILD CRNN ARCHITECTURE (Based on Table 3)
# ============================================================

def build_crnn_model(input_shape=(128, 1024, 1), vocab_size=NUM_CLASSES):
    """
    CRNN Architecture:
    1. CNN: 4 blocks of (Conv3x3 + BatchNorm + Relu + MaxPool2x2)
    2. Reshape: Converts 2D feature maps to 1D sequences
    3. RNN: 2 layers of Bidirectional LSTM
    4. Output: Softmax layer for CTC
    """
    # --- INPUTS ---
    input_img = layers.Input(shape=input_shape, name="image_input")
    
    # --- CNN FEATURE EXTRACTOR ---
    # Block 1: 128x1024 -> 64x512
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 2: 64x512 -> 32x256
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 3: 32x256 -> 16x128
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Block 4: 16x128 -> 8x64
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # --- RESHAPE (MAP TO SEQUENCE) ---
    # After 4 pools, Height is 128/16 = 8. Width is 1024/16 = 64.
    # We transpose to make Width the first dimension (Time Steps)
    # (Batch, 8, 64, 256) -> (Batch, 64, 8, 256)
    x = layers.Permute((2, 1, 3))(x)
    # Flatten Height and Channels: (Batch, 64, 8*256) -> (Batch, 64, 2048)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # --- BIDIRECTIONAL RNN ---
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

    # --- DENSE OUTPUT ---
    # vocab_size includes the <BLANK> token
    y_pred = layers.Dense(vocab_size, activation="softmax", name="softmax_output")(x)

    return models.Model(inputs=input_img, outputs=y_pred, name="OMR_CRNN_Base")

# Create the base model
base_model = build_crnn_model(input_shape=(128, 1024, 1), vocab_size=NUM_CLASSES)
base_model.summary()

# ============================================================
# PHASE 4.2: CTC TRAINING WRAPPER
# ============================================================
MAX_LABEL_LEN = 100   # same as used in padding
# CTC Loss Function
def ctc_loss_lambda(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Inputs for Training
labels = layers.Input(name="the_labels", shape=(MAX_LABEL_LEN,), dtype='int32')
input_length = layers.Input(name='input_length', shape=(1,), dtype='int64')
label_length = layers.Input(name='label_length', shape=(1,), dtype='int64')

# Lambda layer for loss
loss_out = layers.Lambda(ctc_loss_lambda, output_shape=(1,), name='ctc_loss')(
    [base_model.output, labels, input_length, label_length]
)

# Full training model
train_model = models.Model(
    inputs=[base_model.input, labels, input_length, label_length], 
    outputs=loss_out
)

# Compile with dummy loss (since the Lambda layer already calculates the loss)
train_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'ctc_loss': lambda y_true, y_pred: y_pred}
)

# ============================================================
# PHASE 4.3: INFERENCE & DECODING
# ============================================================

def decode_prediction(pred_softmax):
    """
    Decodes the Softmax output into symbol IDs using CTC Greedy Search.
    """
    # Input length is 64 for all samples (1024 width / 16)
    input_len = np.ones(pred_softmax.shape[0]) * pred_softmax.shape[1]
    
    # Decode
    decoded, _ = K.ctc_decode(pred_softmax, input_length=input_len, greedy=True)
    
    # Convert to list of lists
    result = decoded[0].numpy()
    
    return result

# ============================================================
# 🔍 VISUALIZATION: WHAT THE CNN SEES
# ============================================================

def visualize_cnn_features(model, sample_img):
    # Create a model that outputs the first Conv2D layer
    feature_extractor = models.Model(inputs=model.input, outputs=model.layers[1].output)
    features = feature_extractor.predict(sample_img.reshape(1, 128, 1024, 1))
    
    plt.figure(figsize=(15, 5))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(features[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.suptitle("CNN Visual Features (First Layer)")
    plt.show()
# ============================================================
# BUILD TRAIN TENSOR SET (FIX)
# ============================================================

X_train_final = []

limit = min(100, len(train_data))   # keep small for now

for i in range(limit):

    img, label = train_data[i]

    # Step 1: binarize
    binary = binarize_image(img)

    # Step 2: remove staff
    _, no_staff = remove_staff_lines(binary)

    # Step 3: extract strips
    strips, _ = crop_melody_strips(no_staff)

    # Step 4: convert each strip to tensor
    for strip in strips:
        tensor = prepare_tensor_for_model(strip)
        X_train_final.append(tensor)

# Convert to numpy
X_train_final = np.array(X_train_final)

print(f"Training tensor shape: {X_train_final.shape}")

# Run visualization on a training sample
visualize_cnn_features(base_model, X_train_final[0])

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Required for the .meta/.data files
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD THE OFFICIAL VOCABULARY (FIXED)
# ============================================================

VOCAB_FILE_PATH = "vocabulary_agnostic.txt" 

def load_vocab_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}! Please create it with the full list of symbols.")
    with open(path, 'r') as f:
        # splitlines() handles hidden \n characters correctly
        vocab = f.read().splitlines()
    return [v.strip() for v in vocab if v.strip()] # Remove any empty lines

# Load the list
agnostic_vocab = load_vocab_from_file(VOCAB_FILE_PATH)

# Recalibrate mappings - ensure keys are standard Python integers
token_to_idx = {word: int(i) for i, word in enumerate(agnostic_vocab)}
idx_to_token = {int(i): word for i, word in enumerate(agnostic_vocab)}

# CTC Blank Index (The Null Label)
# Standard PrIMuS model puts Blank at the end
BLANK_INDEX = len(agnostic_vocab)
idx_to_token[BLANK_INDEX] = "<BLANK>"

print(f"✅ Vocabulary loaded: {len(agnostic_vocab)} symbols.")
print(f"Index 65 corresponds to: {idx_to_token.get(65, 'STILL MISSING!')}")

# ============================================================
# 2. LOAD THE PRE-TRAINED TENSORFLOW V1 MODEL
# ============================================
# Since you have .meta, .index, and .data files, we use the TF1 Saver.

# Clear the graph and start a session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# Path to your weights (prefix only, as shown in your folder screenshot)
WEIGHTS_PATH = "agnostic_model" 

print("Loading pre-trained weights...")
# 1. Import the graph (the skeleton)
saver = tf.train.import_meta_graph(f"{WEIGHTS_PATH}.meta")
# 2. Restore the values (the brain)
saver.restore(sess, WEIGHTS_PATH)

# Identify the input and output tensors from the saved graph
graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name("model_input:0")
seq_len_tensor = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
logits = tf.get_collection("logits")[0]

# Define the decoder within the graph
decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len_tensor)

print("✅ Model fully restored and ready for prediction.")

# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_music(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Image not found at {image_path}")
    
    # 1. Resize/Normalize (128px height)
    h, w = img.shape
    new_w = int(128 * w / h)
    img_resized = cv2.resize(img, (new_w, 128))
    img_norm = (255. - img_resized) / 255.
    img_input = img_norm.reshape(1, 128, new_w, 1)
    
    # 2. Sequence Length (Width / 16 reduction)
    seq_len = [img_input.shape[2] / 16]

    # 3. Run Inference
    prediction = sess.run(decoded, feed_dict={
        input_tensor: img_input,
        seq_len_tensor: seq_len,
        rnn_keep_prob: 1.0 
    })

    # 4. Extract IDs from SparseTensor
    indices = prediction[0].indices
    values = prediction[0].values
    
    # Extract IDs for the first image in batch
    predicted_ids = [val for idx, val in zip(indices, values) if idx[0] == 0]
    
    # 5. Map IDs to strings with SAFETY check
    predicted_tokens = []
    for tid in predicted_ids:
        clean_id = int(tid) # Convert np.int64 to Python int
        if clean_id in idx_to_token:
            predicted_tokens.append(idx_to_token[clean_id])
        else:
            predicted_tokens.append(f"[UNKNOWN_ID_{clean_id}]")
    
    return predicted_tokens
# ============================================================
# 4. TEST ON A REAL IMAGE
# ============================================================

TEST_IMAGE_PATH = "Corpus/000051652-1_2_1/000051652-1_2_1.png" # Example path
result = predict_music(TEST_IMAGE_PATH)

print("\n--- AI TRANSCRIPTION RESULT ---")
print(" | ".join(result))

# Visualization
test_img = cv2.imread(TEST_IMAGE_PATH)
plt.figure(figsize=(15, 3))
plt.imshow(test_img)
plt.title("Transcribed Music Staff")
plt.axis('off')
plt.show()

import numpy as np

# ============================================================
# 1. LEVENSHTEIN DISTANCE FUNCTION
# ============================================================
def edit_distance(s1, s2):
    """Calculates edit distance between two sequences (lists or strings)."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# ============================================================
# 2. EVALUATION LOOP
# ============================================================

def run_full_evaluation(test_samples):
    total_ser_dist = 0
    total_cer_dist = 0
    total_symbols = 0
    total_chars = 0
    
    num_eval = len(test_samples)
    print(f"Starting evaluation on {num_eval} samples...")

    for i, (img, true_labels) in enumerate(test_samples):
        # 1. Image preprocessing (consistent with our predict function)
        h, w = img.shape
        new_w = int(128 * w / h)
        img_resized = cv2.resize(img, (new_w, 128))
        img_norm = (255. - img_resized) / 255.
        img_input = img_norm.reshape(1, 128, new_w, 1)
        
        # 2. Prediction
        seq_len = [img_input.shape[2] / 16]
        prediction = sess.run(decoded, feed_dict={
            input_tensor: img_input,
            seq_len_tensor: seq_len,
            rnn_keep_prob: 1.0 
        })
        
        # 3. Process Sparse Result
        indices = prediction[0].indices
        values = prediction[0].values
        pred_ids = [int(val) for idx, val in zip(indices, values) if idx[0] == 0]
        pred_labels = [idx_to_token.get(tid, "") for tid in pred_ids]

        # --- SER (Symbol Level) ---
        s_dist = edit_distance(true_labels, pred_labels)
        total_ser_dist += s_dist
        total_symbols += len(true_labels)

        # --- CER (Character Level) ---
        true_str = "".join(true_labels)
        pred_str = "".join(pred_labels)
        c_dist = edit_distance(list(true_str), list(pred_str))
        total_cer_dist += c_dist
        total_chars += len(true_str)

        if i % 10 == 0:
            print(f"Processed {i}/{num_eval}...")

    # Final calculations
    ser = (total_ser_dist / total_symbols) * 100
    cer = (total_cer_dist / total_chars) * 100
    
    return ser, cer

# ============================================================
# 3. RUN EVALUATION
# ============================================================

# Use the 'val_data' split we created earlier (e.g., 50 samples)
test_subset = val_data[:50] 

ser_result, cer_result = run_full_evaluation(test_subset)

print("\n" + "="*45)
print(f"📊 FINAL SYSTEM METRICS (Pre-trained Model)")
print("="*45)
print(f"SER (Symbol Error Rate):    {ser_result:.2f}%")
print(f"LER (Label Error Rate):     {ser_result:.2f}%")
print(f"CER (Character Error Rate): {cer_result:.2f}%")
print("="*45)
print("Interpretation: 0% is perfect. Anything < 5% is excellent.")


import random

def predict_and_compare(sample_data, model_session, idx_to_token):
    """
    sample_data: a single tuple (image_array, true_label_list)
    """
    raw_img, true_labels = sample_data
    
    # 1. Preprocess the image (128px height, aspect ratio preserved)
    h, w = raw_img.shape
    new_w = int(128 * w / h)
    img_resized = cv2.resize(raw_img, (new_w, 128))
    img_norm = (255. - img_resized) / 255.
    img_input = img_norm.reshape(1, 128, new_w, 1)
    
    # 2. Run AI Inference
    seq_len = [img_input.shape[2] / 16]
    prediction = model_session.run(decoded, feed_dict={
        input_tensor: img_input,
        seq_len_tensor: seq_len,
        rnn_keep_prob: 1.0 
    })
    
    # 3. Decode result
    indices = prediction[0].indices
    values = prediction[0].values
    pred_ids = [int(val) for idx, val in zip(indices, values) if idx[0] == 0]
    pred_labels = [idx_to_token.get(tid, f"UNK_{tid}") for tid in pred_ids]

    # 4. Calculate Accuracy (1 - SER)
    # Using the edit_distance function defined in the previous step
    distance = edit_distance(true_labels, pred_labels)
    # Accuracy is based on the number of correct symbols vs total symbols
    # Note: Accuracy can be negative if there are more errors than notes, so we clamp it at 0.
    sample_ser = distance / max(len(true_labels), 1)
    accuracy = max(0, (1 - sample_ser) * 100)

    # 5. Print Results
    print("\n" + "="*60)
    print("🎼 INDIVIDUAL IMAGE PREDICTION")
    print("="*60)
    print(f"ACTUAL NOTES:    {' | '.join(true_labels)}")
    print("-" * 60)
    print(f"PREDICTED NOTES: {' | '.join(pred_labels)}")
    print("="*60)
    print(f"SYMBOL ERRORS:   {distance}")
    print(f"SAMPLE ACCURACY: {accuracy:.2f}%")
    print("="*60)

    # 6. Show the image
    plt.figure(figsize=(15, 3))
    plt.imshow(raw_img, cmap='gray')
    plt.title(f"Processed Image (Accuracy: {accuracy:.2f}%)")
    plt.axis('off')
    plt.show()

# --- EXECUTION ---
# Pick a random sample from your validation data
random_sample = random.choice(val_data)
predict_and_compare(random_sample, sess, idx_to_token)

# ============================================================
# REQUIRED HELPERS — copied here so ablation cell is self-contained
# ============================================================

def segment_music_symbols(img):
    debug = {}
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug["binary"] = binary.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    staff = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    no_staff = cv2.subtract(binary, staff)
    debug["no_staff"] = no_staff.copy()
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(no_staff, cv2.MORPH_OPEN, kernel)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(clean, connect_kernel, iterations=1)
    debug["dilated"] = dilated.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes = []
    H, W = img.shape
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 50:        continue
        if w < 5 or h < 8:  continue
        if w > 0.4 * W:     continue
        if h > 0.9 * H:     continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes, debug


IMG_SIZE = 64

def resize_with_padding(symbol_img, size=IMG_SIZE):
    h, w = symbol_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.float32)
    scale  = size / max(h, w)
    new_h  = max(1, int(h * scale))
    new_w  = max(1, int(w * scale))
    resized = cv2.resize(symbol_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((size, size), dtype=np.uint8)
    y_off   = (size - new_h) // 2
    x_off   = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas.astype(np.float32) / 255.0


def crop_and_pair(original_img, boxes, label_seq):
    grey    = original_img if len(original_img.shape) == 2 \
              else cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    patches_out, labels_out = [], []
    min_len = min(len(boxes), len(label_seq))
    for i in range(min_len):
        x, y, w, h = boxes[i]
        H, W = grey.shape
        crop = grey[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
        if crop.size == 0:
            continue
        patches_out.append(resize_with_padding(crop))
        labels_out.append(label_seq[i])
    return patches_out, labels_out


def encode_label(label_seq, token_to_idx):
    # safely handle vocabs built with or without explicit <UNK>
    unk_id = token_to_idx.get("<UNK>", token_to_idx.get("<BLANK>", 0))
    return [token_to_idx.get(sym, unk_id) for sym in label_seq]

# !pip install miditoolkit 
# or use a simple mapping

def export_to_midi(pred_tokens, filename="output.mid"):
    """
    A simple mapper to turn Agnostic strings into MIDI notes.
    Assumes Treble Clef for this example.
    """
    # Mapping for Agnostic positions to MIDI Pitch
    # L = Line, S = Space (e.g., L2 in Treble is G4 which is MIDI 67)
    pitch_map = {
        "L1": 64, "S1": 65, "L2": 67, "S2": 69, "L3": 71, "S3": 72, "L4": 74, "S4": 76, "L5": 77
    }
    
    midi_sequence = []
    for token in pred_tokens:
        if "note" in token:
            pos = token.split('-')[-1] # Extract L2 or S3
            pitch = pitch_map.get(pos, 60) # Default to Middle C
            midi_sequence.append(pitch)
            
    print(f"Generated MIDI Sequence: {midi_sequence}")
    return midi_sequence

# Example use:
my_music = export_to_midi(result)

'''def plot_learning_curves(history):
    """
    Plots the Training vs Validation Loss to detect overfitting.
    Place this right after history = train_model.fit(...)
    """
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training CTC Loss', color='#1f77b4', lw=2)
    plt.plot(history.history['val_loss'], label='Validation CTC Loss', color='#ff7f0e', lw=2)
    plt.title('Model Loss (Convergence)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # If you tracked SER during training, plot it here. 
    # Otherwise, we plot the learning rate if using ReduceLROnPlateau
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate', color='#2ca02c', lw=2)
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usage:
plot_learning_curves(history)

'''
