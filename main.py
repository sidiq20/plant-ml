import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np 
import joblib
from PIL import Image
from io import BytesIO
import cv2
import json 
import tensorflow as tf

# === load tf model ===
MODEL_PATH = "saved_model/plant_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# === load class indices and label encoder ===
with open("saved_model/class_indecies.json", "r") as f:
    class_indices = json.load(f)
    
idx_to_label = {v: k for k, v in class_indices.items()}

app = FastAPI()

IMG_SIZE = 224  # same image size used during training

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image) / 255.0  # normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image_tensor = preprocess_image(content)
    predictions = model.predict(image_tensor)[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    label = idx_to_label[predicted_idx]
    
    return {
        "prediction": label,
        "confidence": round(confidence, 2),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)