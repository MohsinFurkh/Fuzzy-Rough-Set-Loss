"""
Configuration settings for the segmentation project.
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Model parameters
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = 1

# Training parameters
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
PATIENCE = 10
N_SPLITS = 5

# Loss function parameters
ALPHA = 0.5  # For combined loss
GAMMA = 2.0  # For focal loss
ALPHA_FOCAL = 0.25  # For focal loss
TVERSKY_ALPHA = 0.3  # For Tversky loss
TVERSKY_BETA = 0.7   # For Tversky loss

# Evaluation parameters
TOLERANCE = 2  # For boundary F-score calculation
SIZE_THRESHOLD = 100  # For small lesion detection
NUM_MC_SAMPLES = 20   # For Monte Carlo dropout

# Visualization parameters
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 8)
COLOR_MAP = 'viridis'
