import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from utils.preprocessing import preprocess_image

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="ECG Classification Fusion CNN",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Klasifikasi Sinyal ECG")
st.caption("VGG19 + DenseNet201 + InceptionV3 | Decision Fusion")

CLASS_NAMES = ["H2O", "H2O+NaOH"]

# ===============================
# Load Model (Cache)
# ===============================
@st.cache_resource
def load_models():
    vgg = tf.keras.models.load_model("models/vgg19_best_model.h5")
    dense = tf.keras.models.load_model("models/densenet201_best_model.h5")
    inc = tf.keras.models.load_model("models/inceptionv3_best_model.h5")
    return vgg, dense, inc

with st.spinner("🔄 Memuat model..."):
    vgg_model, dense_model, inc_model = load_models()

st.success("✅ Model berhasil dimuat")

# ===============================
# Upload Gambar
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload citra ECG",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra ECG", use_column_width=True)

    img_np = np.array(image)

    # ===============================
    # Preprocessing
    # ===============================
    img_vgg = preprocess_image(img_np, "vgg")
    img_dense = preprocess_image(img_np, "densenet")
    img_inc = preprocess_image(img_np, "inception")

    # ===============================
    # Prediksi
    # ===============================
    with st.spinner("🧠 Menganalisis..."):
        pred_vgg = vgg_model.predict(img_vgg)[0][0]
        pred_dense = dense_model.predict(img_dense)[0][0]
        pred_inc = inc_model.predict(img_inc)[0][0]

        # Equal Weight Fusion (SAMA dengan Colab)
        fusion_prob = (pred_vgg + pred_dense + pred_inc) / 3
        fusion_label = CLASS_NAMES[int(fusion_prob > 0.5)]

    # ===============================
    # Output
    # ===============================
    st.subheader("📊 Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fusion Prediction", fusion_label)

    with col2:
        st.metric("Confidence", f"{fusion_prob:.4f}")

    st.markdown("---")
    st.write("### 🔍 Probabilitas Model Individu")
    st.write(f"- **VGG19**       : {pred_vgg:.4f}")
    st.write(f"- **DenseNet201**: {pred_dense:.4f}")
    st.write(f"- **InceptionV3**: {pred_inc:.4f}")

    st.info("📌 Threshold klasifikasi = 0.5 (sesuai Colab)")
