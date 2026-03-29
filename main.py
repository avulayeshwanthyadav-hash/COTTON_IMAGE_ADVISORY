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
import json
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")

MODEL_PATH = "model/cotton_model.h5"
CSV_PATH = "cotton_advisory_image.csv"
CLASS_PATH = "model/classes.json"

DEFAULT_LANGUAGE = "en"
# ---------------------------------------

app = FastAPI()

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class mapping 🔥
with open(CLASS_PATH) as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())

# Load CSV
data = pd.read_csv(CSV_PATH, encoding="utf-8")
data.columns = data.columns.str.strip()
data["disease"] = data["disease"].str.strip()
# 🔥 Fix model label mismatches
disease_mapping = {
    "curl_virus": "leaf_curl",
    "fussarium_wilt": "fusarium_wilt"
}

# Language store
user_language_store = {}

def predict_disease(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    confidence = np.max(preds)
    index = np.argmax(preds)

    return classes[index], confidence


@app.post("/whatsapp")
async def whatsapp_bot(request: Request):
    form = await request.form()
    media_url = form.get("MediaUrl0")
    user_msg = form.get("Body", "").strip().lower()
    user_number = form.get("From")

    resp = MessagingResponse()

    # -------- LANGUAGE --------
    if user_msg in ["hi", "hindi"]:
        user_language_store[user_number] = "hi"
        resp.message("✅ भाषा सेट हो गई (Hindi)")
        return Response(str(resp), media_type="application/xml")

    elif user_msg in ["te", "telugu"]:
        user_language_store[user_number] = "te"
        resp.message("✅ భాష సెట్ చేయబడింది (Telugu)")
        return Response(str(resp), media_type="application/xml")

    elif user_msg in ["en", "english"]:
        user_language_store[user_number] = "en"
        resp.message("✅ Language set to English")
        return Response(str(resp), media_type="application/xml")

    language = user_language_store.get(user_number, DEFAULT_LANGUAGE)

    # -------- NO IMAGE --------
    if not media_url:
        resp.message(
            "📸 Please upload a cotton leaf photo.\n\n"
            "EN / HI / TE"
        )
        return Response(str(resp), media_type="application/xml")

    # -------- DOWNLOAD IMAGE --------
    response = requests.get(media_url,
                            auth=HTTPBasicAuth(TWILIO_SID, TWILIO_AUTH))

    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        resp.message("❌ Invalid image. Send clear cotton leaf photo.")
        return Response(str(resp), media_type="application/xml")

    # -------- PREDICT --------
    disease, confidence = predict_disease(img)
    disease = disease_mapping.get(disease, disease)
    # ❗ Confidence check
    if confidence < 0.7:
        resp.message("❗ Unable to detect clearly. Please send a clearer image.")
        return Response(str(resp), media_type="application/xml")

    # -------- GET DATA --------
    row = data[data["disease"] == disease]

    if not row.empty:
        advice = row.iloc[0].get(language)

        if pd.isna(advice):
            advice = row.iloc[0].get("en")

        disease_te = row.iloc[0].get("te_name", disease)
        disease_hi = row.iloc[0].get("hi_name", disease)
    else:
        advice = "Advisory not available."
        disease_te = disease
        disease_hi = disease

    # -------- RESPONSE --------
    if language == "te":
        message = f"✅ వ్యాధి గుర్తించబడింది: {disease_te}\n\n{advice}"

    elif language == "hi":
        message = f"✅ रोग की पहचान: {disease_hi}\n\n{advice}"

    else:
        message = f"✅ Disease Detected: {disease}\n\n{advice}"

    resp.message(message)
    return Response(str(resp), media_type="application/xml")
