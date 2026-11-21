"""Inference Engine for Fruit Quality Grading.

Accepts an input fruit image (BGR array or image path), executes the OpenCV
preprocessing pipeline, extracts physical features, and performs inference
using the hybrid deep neural network.

Usage (CLI):
  python inference.py path/to/fruit.jpg
"""
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import joblib
import tensorflow as tf

import config
from preprocessing import preprocess, load_image
from features import extract_features, summarize_features


class FruitGrader:
    """Predictor class for Fruit Quality Grading."""

    def __init__(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model not found at {config.MODEL_PATH}. Train first using 'python train.py'.")

        self.classes = json.load(open(config.CLASSES_PATH))
        self.cfg = json.load(open(config.MODEL_CFG_PATH))
        self.scaler = joblib.load(config.SCALER_PATH)
        self.model = tf.keras.models.load_model(config.MODEL_PATH)

    def predict(self, bgr):
        """Run complete grading pipeline on input BGR image.

        Returns:
            dict with predicted grade, confidence, probabilities,
            OpenCV processed visuals, and diagnostic features.
        """
        out = preprocess(bgr)
        feats = extract_features(out["rgb"], out["mask"])
        feats_scaled = self.scaler.transform(feats[None, :])

        img_tensor = out["rgb"][None, ...].astype(np.float32)
        model_input = [img_tensor, feats_scaled] if self.cfg["use_features"] else img_tensor

        proba = self.model.predict(model_input, verbose=0)[0]
        top_idx = int(np.argmax(proba))
        grade_key = self.classes[top_idx]
        conf = float(proba[top_idx])

        diag = summarize_features(feats)
        rec = config.SHELF_LIFE_RECOMMENDATIONS.get(grade_key, {})

        # Prepare visual overlay with label banner
        overlay = out["overlay"].copy()
        label_text = f"{config.GRADE_LABELS.get(grade_key, grade_key)}: {conf:.1%}"
        bx, by, bw, bh = out["bbox"]

        # Place banner above bounding box
        text_y = max(25, by - 10)
        cv2.putText(overlay, label_text, (bx, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        return {
            "grade": grade_key,
            "label": config.GRADE_LABELS.get(grade_key, grade_key),
            "confidence": conf,
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
            "diagnostics": diag,
            "recommendation": rec,
            "processed": out,
            "features": feats,
            "annotated_overlay": overlay
        }


def main():
    parser = argparse.ArgumentParser(description="Fruit Quality Grading CLI Predictor")
    parser.add_argument("image", help="Path to input fruit image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print(f"\nAnalyzing: {img_path.name} ...")
    grader = FruitGrader()
    bgr = load_image(img_path)
    result = grader.predict(bgr)

    print("\n" + "=" * 55)
    print("FRUIT QUALITY GRADING REPORT")
    print("=" * 55)
    print(f"Predicted Grade:  {result['label']}")
    print(f"Confidence Score: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for c, p in result["probabilities"].items():
        print(f"  {c:<8s}: {p:6.2%}")

    d = result["diagnostics"]
    print("\nOpenCV Diagnostic Features:")
    print(f"  Defect Severity:   {d['defect_severity_pct']:.1f}%")
    print(f"  Browning Index:    {d['browning_index']:.1f}%")
    print(f"  Dark Decay Ratio:  {d['dark_decay_ratio']:.1f}%")
    print(f"  Texture Roughness: {d['texture_roughness']:.2f}")
    print(f"  Shape Circularity: {d['shape_circularity']:.2f}")

    r = result["recommendation"]
    print(f"\nCommercial Recommendation:")
    print(f"  Status:          {r.get('status', 'N/A')}")
    print(f"  Action:          {r.get('action', 'N/A')}")
    print(f"  Est. Shelf Life: {r.get('est_shelf_life', 'N/A')}")

    out_file = config.OUTPUT_DIR / "prediction.jpg"
    cv2.imwrite(str(out_file), result["annotated_overlay"])
    print(f"\nSaved annotated image -> {out_file}")
    print("=" * 55)


if __name__ == "__main__":
    main()
