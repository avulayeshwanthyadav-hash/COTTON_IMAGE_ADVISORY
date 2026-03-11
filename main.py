from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from requests.auth import HTTPBasicAuth

import tensorflow as tf
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

# ------------------- CONFIG -------------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")

MODEL_PATH = "model/cotton_model.h5"
CSV_PATH = "cotton_advisory_image.csv"

DEFAULT_LANGUAGE = "en"  # default language en / hi / te
# ----------------------------------------------

app = FastAPI()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load CSV
data = pd.read_csv(CSV_PATH)

# Class order must match train_data.class_indices
classes = ['bacterial_blight', 'fusarium_wilt', 'healthy', 'leaf_curl']

def predict_disease(img: Image.Image) -> str:
    img = img.resize((224, 224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    return classes[np.argmax(preds)]

@app.post("/whatsapp")
async def whatsapp_bot(request: Request):
    form = await request.form()
    media_url = form.get("MediaUrl0")
    user_msg = form.get("Body", "").strip().lower()

    resp = MessagingResponse()

    # Language selection
    if user_msg in ["hi", "hindi"]:
        language = "hi"
    elif user_msg in ["te", "telugu"]:
        language = "te"
    else:
        language = DEFAULT_LANGUAGE

    if not media_url:
        resp.message(
            "📸 Please upload a cotton leaf photo.\n\n"
            "Reply with:\nEN – English\nHI – Hindi\nTE – Telugu"
        )
        return Response(str(resp), media_type="application/xml")

    # Download image
    response = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_SID, TWILIO_AUTH))
    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        resp.message("❌ Unable to read image. Send a clear cotton leaf photo.")
        return Response(str(resp), media_type="application/xml")

    disease = predict_disease(img)

    try:
        advice = data[data["disease"]==disease][language].values[0]
    except:
        advice = "Advisory not available."

    resp.message(f"✅ Disease Detected: {disease}\n\n{advice}")
    return Response(str(resp), media_type="application/xml")

