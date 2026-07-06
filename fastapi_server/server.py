import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
print(ROOT)
print(sys.path)
print("--------------------------------------")
from PIL import Image
import io

# NEW FIXED: Now importing from the neutral core directory instead of the streamlit app folder
from core.model_utils import classify_car_damage
from fastapi import FastAPI, File, UploadFile

print("--------------------------------------")
app = FastAPI()

@app.post("/predict-damage")
async def predict_damage(file: UploadFile = File(...)):
    try:
        # raw = await file.read()
        raw = file.file.read()
        raw_bytes = io.BytesIO(raw)
        img = Image.open(raw_bytes)

        prediction = classify_car_damage(img)

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
