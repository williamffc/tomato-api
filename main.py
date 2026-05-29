from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

# ✅ CORS para que tu frontend pueda llamar la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo una sola vez al iniciar
model = tf.keras.models.load_model("tomato_model.keras")
class_names = ["dañado", "viejo", "verde", "maduro"]

@app.get("/")
def root():
    return {"status": "TomatoScan API corriendo 🍅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer imagen
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesar — SIN dividir por 255 (Rescaling está en el modelo)
    img_input = np.expand_dims(
        cv2.resize(img_rgb, (64, 64)), axis=0
    ).astype(np.float32)

    # Predecir
    pred  = model.predict(img_input, verbose=0)[0]
    idx   = int(np.argmax(pred))
    label = class_names[idx]
    conf  = float(pred[idx])

    return {
        "label": label,
        "confidence": round(conf * 100, 2),
        "scores": {
            class_names[i]: round(float(pred[i]) * 100, 2)
            for i in range(len(class_names))
        }
    }