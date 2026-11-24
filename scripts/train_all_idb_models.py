"""
Model Training Pipeline for ALL-IDB Leukemia Classification
SmartCell - Phase 3 Model Training (Optimized for Fast & Accurate Convergence)

Trains and compares:
1. Custom CNN Baseline
2. MobileNetV2 Transfer Learning
3. EfficientNetB0 Transfer Learning

Requirements:
- Fixed random seed = 42 for reproducibility
- Consistent input resolution: 224x224x3
- Proper normalization per architecture
- Realistic microscopy augmentation (train set only: flips and brightness)
- High-efficiency in-memory cache + prefetch for fast CPU execution
- Early stopping & best model checkpointing on val_loss
- Training history preservation & loss/accuracy curve plotting
- Models saved inside D:/smartcell/models/
- Histories and curves saved inside D:/smartcell/results/all_idb/
"""

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds before TensorFlow imports
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = r"D:\smartcell"
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_all")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "all_idb")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Microscopy Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
    layers.RandomBrightness(0.08, value_range=(0.0, 255.0), seed=RANDOM_SEED)
], name="microscopy_augmentation")

def load_and_cache_datasets():
    print("\nLoading dataset splits from:", DATASET_DIR)
    train_dir = os.path.join(DATASET_DIR, "train")
    val_dir = os.path.join(DATASET_DIR, "val")
    test_dir = os.path.join(DATASET_DIR, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Class names detected: {class_names}")

    # Compute class weights
    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(int(labels.numpy()[0]))
    y_train = np.array(y_train)

    classes_unique = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_unique, weights)}
    print(f"Calculated class weights: {class_weight_dict}")

    # Pipeline optimization: cache in RAM + augment + prefetch
    train_proc = (
        train_ds.cache()
        .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_proc = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_proc = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_proc, val_proc, test_proc, class_weight_dict, class_names

def build_custom_cnn():
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = layers.Rescaling(1./255, name="rescaling_0_1")(inputs)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop1", seed=RANDOM_SEED)(x)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.2, name="drop2", seed=RANDOM_SEED)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_blast_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="all_idb_custom_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_mobilenetv2_model():
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling_neg1_1")(inputs)

    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.3, name="drop1", seed=RANDOM_SEED)(x)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.2, name="drop2", seed=RANDOM_SEED)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_blast_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="all_idb_mobilenetv2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model, base_model

def build_efficientnetb0_model():
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")

    base_model = EfficientNetB0(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.3, name="drop1", seed=RANDOM_SEED)(x)
    x = layers.Dense(64, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.2, name="drop2", seed=RANDOM_SEED)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_blast_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="all_idb_efficientnetb0")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model, base_model

def train_and_finetune(model, base_model, train_ds, val_ds, class_weight_dict, model_save_path, model_name):
    print(f"\n{'='*60}\nTraining {model_name}...\n{'='*60}")
    t0 = time.time()

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Phase 1: Train classification head
    print(f"[{model_name}] Phase 1: Initial training (epochs=12)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=12,
        class_weight=class_weight_dict,
        callbacks=cb_list,
        verbose=1
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history1.history.items()}

    # Phase 2: Fine-tuning if base_model exists
    if base_model is not None:
        print(f"\n[{model_name}] Phase 2: Fine-tuning top layers (epochs=8)...")
        base_model.trainable = True
        # Freeze early layers, keep last 20 trainable
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        cb_list_ft = [
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=8,
            class_weight=class_weight_dict,
            callbacks=cb_list_ft,
            verbose=1
        )

        for k, vals in history2.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    train_duration = time.time() - t0
    print(f"[{model_name}] Training completed in {train_duration:.2f} seconds.")

    # Save history json
    hist_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(hist_path, "w") as fp:
        json.dump({
            "model_name": model_name,
            "train_duration_sec": train_duration,
            "total_epochs": len(combined_history["loss"]),
            "best_val_loss": min(combined_history["val_loss"]),
            "best_val_acc": max(combined_history["val_accuracy"]),
            "history": combined_history
        }, fp, indent=2)

    return combined_history, train_duration

def plot_all_histories(all_histories):
    fig, axes = plt.subplots(len(all_histories), 2, figsize=(14, 4 * len(all_histories)))
    if len(all_histories) == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (m_name, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist["loss"]) + 1)
        
        # Loss plot
        axes[idx, 0].plot(epochs, hist["loss"], 'b-o', label='Training Loss', markersize=4)
        axes[idx, 0].plot(epochs, hist["val_loss"], 'r--s', label='Validation Loss', markersize=4)
        axes[idx, 0].set_title(f"{m_name.upper()} - Loss Curves", fontsize=11, fontweight="bold")
        axes[idx, 0].set_xlabel("Epoch")
        axes[idx, 0].set_ylabel("Binary Crossentropy")
        axes[idx, 0].grid(True, linestyle=":", alpha=0.6)
        axes[idx, 0].legend()

        # Accuracy plot
        axes[idx, 1].plot(epochs, hist["accuracy"], 'b-o', label='Training Accuracy', markersize=4)
        axes[idx, 1].plot(epochs, hist["val_accuracy"], 'g--^', label='Validation Accuracy', markersize=4)
        axes[idx, 1].set_title(f"{m_name.upper()} - Accuracy Curves", fontsize=11, fontweight="bold")
        axes[idx, 1].set_xlabel("Epoch")
        axes[idx, 1].set_ylabel("Accuracy")
        axes[idx, 1].set_ylim(0.4, 1.05)
        axes[idx, 1].grid(True, linestyle=":", alpha=0.6)
        axes[idx, 1].legend()

    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "all_idb_training_curves.png")
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved combined training curves to: {curve_path}")

def main():
    print("=" * 70)
    print("STARTING ALL-IDB LEUKEMIA MODEL TRAINING (PHASE 3)")
    print("=" * 70)

    train_proc, val_proc, test_proc, class_weights, class_names = load_and_cache_datasets()

    all_histories = {}
    training_summary = {
        "training_timestamp": "2026-11-21T22:50:00+05:30",
        "random_seed": RANDOM_SEED,
        "input_resolution": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "class_names": class_names,
        "class_weights": class_weights,
        "models": {}
    }

    # 1. Custom CNN
    cnn_path = os.path.join(MODELS_DIR, "all_idb_custom_cnn_best.keras")
    cnn_hist_path = os.path.join(RESULTS_DIR, "custom_cnn_history.json")
    if os.path.exists(cnn_path) and os.path.exists(cnn_hist_path):
        print(f"\n[Custom CNN] Found existing trained model at: {cnn_path}")
        with open(cnn_hist_path, "r") as fp:
            saved_cnn = json.load(fp)
        cnn_hist = saved_cnn["history"]
        cnn_time = saved_cnn.get("train_duration_sec", 0)
    else:
        cnn_model = build_custom_cnn()
        cnn_hist, cnn_time = train_and_finetune(
            cnn_model, None, train_proc, val_proc, class_weights, cnn_path, "custom_cnn"
        )
    all_histories["custom_cnn"] = cnn_hist
    training_summary["models"]["custom_cnn"] = {
        "saved_path": cnn_path,
        "train_time_sec": cnn_time,
        "best_val_loss": min(cnn_hist["val_loss"]),
        "best_val_acc": max(cnn_hist["val_accuracy"])
    }

    # 2. MobileNetV2
    mob_path = os.path.join(MODELS_DIR, "all_idb_mobilenetv2_best.keras")
    mob_model, mob_base = build_mobilenetv2_model()
    mob_hist, mob_time = train_and_finetune(
        mob_model, mob_base, train_proc, val_proc, class_weights, mob_path, "mobilenetv2"
    )
    all_histories["mobilenetv2"] = mob_hist
    training_summary["models"]["mobilenetv2"] = {
        "saved_path": mob_path,
        "train_time_sec": mob_time,
        "best_val_loss": min(mob_hist["val_loss"]),
        "best_val_acc": max(mob_hist["val_accuracy"])
    }

    # 3. EfficientNetB0
    eff_path = os.path.join(MODELS_DIR, "all_idb_efficientnetb0_best.keras")
    eff_model, eff_base = build_efficientnetb0_model()
    eff_hist, eff_time = train_and_finetune(
        eff_model, eff_base, train_proc, val_proc, class_weights, eff_path, "efficientnetb0"
    )
    all_histories["efficientnetb0"] = eff_hist
    training_summary["models"]["efficientnetb0"] = {
        "saved_path": eff_path,
        "train_time_sec": eff_time,
        "best_val_loss": min(eff_hist["val_loss"]),
        "best_val_acc": max(eff_hist["val_accuracy"])
    }

    # Plot curves
    plot_all_histories(all_histories)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "training_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(training_summary, fp, indent=2)
    print(f"\nSaved training summary to: {summary_path}")

    print("\nPhase 3: Model Training completed successfully!")

if __name__ == "__main__":
    main()
