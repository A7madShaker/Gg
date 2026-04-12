import io
import pickle
import torch
import torch.storage
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# ── Fix: load CUDA model on CPU ──────────────────────────────────────────────
def _load_from_bytes_cpu(b):
    return torch.load(io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)

torch.storage._load_from_bytes = _load_from_bytes_cpu
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Normal",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "hypodontia",
    "Cancer",
]

# Load model once at startup
with open("Tooth-clas.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

# Image preprocessing (standard EfficientNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = FastAPI(title="Tooth Classifier API")


@app.get("/")
def root():
    return {"message": "Tooth Classifier is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Please upload an image file"})

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # add batch dim

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()

        return {
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
