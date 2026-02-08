import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown

# ---------------------------
# SETUP MODEL PATH DAN DOWNLOAD
# ---------------------------
os.makedirs("models", exist_ok=True)

# Google Drive file IDs untuk model
VGG19_ID = "YOUR_VGG19_FILE_ID"
DENSE_ID = "YOUR_DENSE_FILE_ID"
INC_ID = "YOUR_INCEPTION_FILE_ID"

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/drive/folders/1kgYrgIK7eyD0bJOmITyKmJO9NRFiqW0O?usp=drive_link"
        st.info(f"Downloading {os.path.basename(output_path)} from Google Drive...")
        gdown.download(url, output_path, quiet=False)
    else:
        st.success(f"{os.path.basename(output_path)} already exists.")

# Download semua model
download_model(VGG19_ID, "models/vgg19_best_model.h5")
download_model(DENSE_ID, "models/dense_best_model.h5")
download_model(INC_ID, "models/inception_best_model.h5")

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_models():
    vgg_model = tf.keras.models.load_model("models/vgg19_best_model.h5")
    dense_model = tf.keras.models.load_model("models/dense_best_model.h5")
    inc_model = tf.keras.models.load_model("models/inception_best_model.h5")
    return vgg_model, dense_model, inc_model

vgg_model, dense_model, inc_model = load_models()

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("Klasifikasi H2O & H2O + NaOH")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Contoh preprocessing
    img_array = np.array(image.resize((224, 224)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi (contoh menggunakan vgg_model)
    pred = vgg_model.predict(img_array)
    st.write("Prediksi VGG19:", pred)
