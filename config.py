# config.py
import os
import torch

# --- GET ROOT ---
import sys
from pathlib import Path
def get_project_root() -> Path:    
    """Traverse up to find the root directory via README.md anchor."""
    # Safely get __file__ from globals without throwing an error if missing    
    file_path = globals().get("__file__")
    
    if file_path:
        current = Path(file_path).resolve().parent  
    else: 
        current = Path.cwd()        

    # Search loop
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists():
            return parent

    # Explicit, loud failure (avoids silent bugs)
    raise FileNotFoundError(
        f"Project root anchor 'README.md' not found in '{current}' or any of its parent directories. "
        f"Please ensure you are running the code inside the project repository."
    ) 

# --- GLOBAL PATHS ---
PROJECT_ROOT = get_project_root()
# CRITICAL: Automatically add root to sys.path so src imports work anywhere
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR     = PROJECT_ROOT / "data/car_damage"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs" 

# Ensure directories exist
# Automatically ensure directories exist on the hard drive
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# --- MODEL HYPERPARAMETERS ---
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.001
IMAGE_SIZE    = (224, 224) # smallest image size is 225, so setting one less & even

# --- REUSABLE CONFIGURATIONS ---
RANDOM_SEED     = 42
SET_NUM_THREADS = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Auto-create directories if they do not exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)
