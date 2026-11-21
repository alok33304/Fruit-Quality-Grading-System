"""Model Training Pipeline for Fruit Quality Grading.

Trains:
  1. Classical Baseline: Random Forest Classifier on OpenCV Handcrafted Features.
  2. Proposed Deep Hybrid Model: MobileNetV2 + Feature Fusion.
     - Phase 1: Train classification head with frozen backbone (learning rate = 1e-3).
     - Phase 2: Fine-tune top 30 layers with small learning rate (1e-5).

Usage:
  python train.py --epochs 10 --fine_tune_epochs 5
"""
import argparse
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import config
from dataset import load_split
from model import build_model, unfreeze_top


def plot_history(histories, path):
    """Plot and save training and validation accuracy and loss curves."""
    acc = sum([h.history.get("accuracy", []) for h in histories], [])
    val_acc = sum([h.history.get("val_accuracy", []) for h in histories], [])
    loss = sum([h.history.get("loss", []) for h in histories], [])
    val_loss = sum([h.history.get("val_loss", []) for h in histories], [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Train Accuracy", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, val_acc, "s-", label="Val Accuracy", color="#ff7f0e", lw=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim([0.5, 1.02])
    ax1.legend(loc="lower right", fontsize=10)

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Train Loss", color="#1f77b4", lw=2)
    ax2.plot(epochs_range, val_loss, "s-", label="Val Loss", color="#ff7f0e", lw=2)
    ax2.set_title("Model Loss Across Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Categorical Crossentropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    # Mark fine-tuning boundary if Phase 2 was run
    if len(histories) > 1 and len(histories[0].history.get("loss", [])) > 0:
        split_pt = len(histories[0].history["loss"]) + 0.5
        for ax in (ax1, ax2):
            ax.axvline(split_pt, color="gray", linestyle=":", lw=1.5, label="Fine-tuning start")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved training curves -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Fruit Quality Grading System")
    parser.add_argument("--data_dir", default=str(config.DATA_DIR), help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Phase 1 head training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Phase 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--no_features", action="store_true", help="Ablation: image-only MobileNetV2")
    parser.add_argument("--no_pretrained", action="store_true", help="Train from scratch without ImageNet")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FRUIT QUALITY GRADING SYSTEM - TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load Preprocessed Datasets
    print("\n[Step 1/5] Loading & Preprocessing Training & Validation Data ...")
    X_tr, F_tr, y_tr, classes = load_split(f"{args.data_dir}/train")
    X_va, F_va, y_va, _ = load_split(f"{args.data_dir}/val", classes=classes)

    print(f"  Training samples:   {len(X_tr)} (Classes: {classes})")
    print(f"  Validation samples: {len(X_va)}")

    # 2. Standardize Handcrafted Features
    print("\n[Step 2/5] Fitting Feature Scaler (StandardScaler) ...")
    scaler = StandardScaler().fit(F_tr)
    F_tr_scaled = scaler.transform(F_tr)
    F_va_scaled = scaler.transform(F_va)

    joblib.dump(scaler, config.SCALER_PATH)
    with open(config.CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    with open(config.MODEL_CFG_PATH, "w") as f:
        json.dump({"use_features": not args.no_features, "classes": classes}, f, indent=2)

    # 3. Train Classical Baseline (Random Forest)
    print("\n[Step 3/5] Training Classical Baseline (Random Forest on OpenCV Features) ...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=config.SEED,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(F_tr, y_tr)
    rf_val_acc = rf.score(F_va, y_va)
    joblib.dump(rf, config.RF_PATH)
    print(f"  Random Forest Baseline Validation Accuracy: {rf_val_acc * 100:.2f}%")

    # 4. Build & Train Hybrid Deep Learning Model
    print("\n[Step 4/5] Constructing Hybrid MobileNetV2 + Feature Fusion Model ...")
    use_feats = not args.no_features
    model = build_model(
        num_classes=len(classes),
        feature_dim=F_tr.shape[1],
        use_features=use_feats,
        weights=None if args.no_pretrained else "imagenet"
    )

    # Class weighting to ensure balanced learning
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)))
    train_inputs = [X_tr, F_tr_scaled] if use_feats else X_tr
    val_inputs = [X_va, F_va_scaled] if use_feats else X_va

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            verbose=1
        )
    ]

    # --- Phase 1: Train Head ---
    print("\n>>> Phase 1: Training Classification Head (Backbone Frozen, LR=1e-3) ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    h1 = model.fit(
        train_inputs,
        y_tr,
        validation_data=(val_inputs, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        callbacks=callbacks
    )
    histories = [h1]

    # --- Phase 2: Fine-Tuning ---
    if args.fine_tune_epochs > 0:
        print("\n>>> Phase 2: Fine-Tuning Top 30 MobileNetV2 Layers (LR=1e-5) ...")
        unfreeze_top(model, n_layers=30)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        h2 = model.fit(
            train_inputs,
            y_tr,
            validation_data=(val_inputs, y_va),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            class_weight=cw,
            callbacks=callbacks
        )
        histories.append(h2)

    # 5. Save Outputs & Curves
    print("\n[Step 5/5] Saving Trained Model and Generating Training Curves ...")
    plot_history(histories, config.OUTPUT_DIR / "training_curves.png")
    model.save(config.MODEL_PATH)

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    print(f"Next step: Run 'python evaluate.py' to test model on held-out test data.")
    print("=" * 65)


if __name__ == "__main__":
    main()
