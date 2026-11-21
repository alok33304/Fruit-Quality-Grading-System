"""Central Configuration for Fruit Quality Grading System.

All system paths, image dimensions, and hyperparameters are defined here.
Designed to be clean, simple, and student-friendly.
"""
from pathlib import Path

# Base project directories
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Image & Training Hyperparameters
IMG_SIZE = 224              # Resolution for MobileNetV2 input (224x224x3)
BATCH_SIZE = 32            # Mini-batch size
SEED = 42                  # Random seed for reproducibility

# Saved Model & Artifact Paths
MODEL_PATH = MODEL_DIR / "fruit_grader.keras"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
CLASSES_PATH = MODEL_DIR / "classes.json"
RF_PATH = MODEL_DIR / "rf_baseline.joblib"
MODEL_CFG_PATH = MODEL_DIR / "model_cfg.json"

# Human-readable quality labels and commercial grading descriptions
GRADE_LABELS = {
    "fresh": "Grade A (Fresh / Good Quality)",
    "medium": "Grade B (Medium / Table Ripe Quality)",
    "rotten": "Grade C (Rotten / Bad / Reject Quality)",
}

# Commercial Shelf-Life & Handling Recommendations
SHELF_LIFE_RECOMMENDATIONS = {
    "fresh": {
        "status": "Premium Quality",
        "action": "Suitable for export and premium grocery shelves.",
        "est_shelf_life": "10 to 21 days (refrigerated)",
        "color": "#28a745"  # Green
    },
    "medium": {
        "status": "Commercial Table Ripe",
        "action": "Suitable for immediate local retail sale or fresh juice processing.",
        "est_shelf_life": "2 to 5 days",
        "color": "#fd7e14"  # Orange
    },
    "rotten": {
        "status": "Spoiled / Defective",
        "action": "Reject from edible chain; divert to composting or biomass.",
        "est_shelf_life": "0 to 1 day (Immediate discard)",
        "color": "#dc3545"  # Red
    }
}
