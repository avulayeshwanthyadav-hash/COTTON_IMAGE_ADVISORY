import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_DIR = "dataset/train"  # your dataset folder
MODEL_PATH = "model/cotton_model.h5"
# ----------------------------------------

# Augmented training data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Base model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base.output)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(base.input, output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

print("Class indices:", train_data.class_indices)  # important to match later

# Train
model.fit(train_data, epochs=EPOCHS)

# Save
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print("Model saved at", MODEL_PATH)
