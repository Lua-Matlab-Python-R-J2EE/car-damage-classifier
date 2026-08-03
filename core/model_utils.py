import sys
from pathlib import Path

# Setup paths cleanly for local and cloud imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
print(ROOT)
print(sys.path)
print("--------------------------------------")

from PIL import Image
from src.helpers import val_test_transform
import config
import torch
import torchvision.models as models
import torch.nn as nn

print("--------------------------------------")
# Define target class labels (adjusted to match exact classes)
CLASS_NAMES = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
print("--------------------------------------")

# Create a private variable to hold the model instance in memory
_CACHED_MODEL = None

def load_production_model() -> torch.nn.Module:
    """
    Initialises the transfer learning architecture and loads tuned weights.
    Caches the model globally so it only loads once.
    """
    global _CACHED_MODEL
    
    # If the model is already loaded, return it immediately without reloading
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    # 1. Start with the completely empty baseline backbone structure
    model = models.mobilenet_v3_small(weights=None)

    # 2. Redesign the complete head EXACTLY how it was done during training
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=len(CLASS_NAMES), bias=True)
    )

    # 3. Use an absolute path calculation to find the weights safely from the root directory
    weights_path = ROOT / "streamlit_app" / "model" / "mobilenet_v3_small_custom_best.pth"

    # Load tuned weights, map to available hardware, and set model to evaluation mode
    state_dict = torch.load(str(weights_path), map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # dropout/batch_norm off

    # Save the loaded model to our cache variable
    _CACHED_MODEL = model.to(config.DEVICE)
    return _CACHED_MODEL


# Load the model once globally at server startup so the application remains fast
PROD_MODEL = load_production_model()


def classify_car_damage(pil_image: Image.Image) -> str:
    """
    Accepts a raw PIL image, applies the official helper transformation,
    and returns the model prediction.
    """
    print(f"Size of input image: {pil_image.size}")

    # 1. Transform the single PIL image into a single 3D tensor
    transformed_image = val_test_transform(pil_image)
    print(f"Size of transformed image: {transformed_image.shape}")

    # 2. Now expand the dimension to create a batch of 1 for the PyTorch model
    prepared_input = transformed_image.unsqueeze(0).to(config.DEVICE)
    print(f"Size of prepared_input image: {prepared_input.shape}")

    # 3. Keep layers/weights frozen - run inference using the global cached model
    with torch.no_grad(): # inference phase; no backpropagation needed
        outputs = PROD_MODEL(prepared_input)
        prediction_index = torch.argmax(outputs, dim=1) # Directly pull the highest index class

    # Extract the single integer value from the tensor
    final_class_id = prediction_index.item()

    return CLASS_NAMES[final_class_id]
