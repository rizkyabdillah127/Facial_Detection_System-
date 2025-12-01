import streamlit as st
import cv2
import numpy as np
from utils.inference import load_model, predict_identity
# Tidak perlu import PIL karena kita menggunakan cv2.imdecode

st.set_page_config(page_title="Face Recognition", layout="centered")

st.title("📷 Face Recognition App")
st.write("Silakan pilih: Ambil foto langsung dari kamera atau unggah file foto dari perangkat Anda.")

# =============================
# 1. Load Model (sekali saja)
# =============================
model = load_model()

if model is None:
    st.error("Model gagal dimuat. Silakan periksa file 'model/model_efficientnet.pth' dan dependensi PyTorch Anda.")
    st.stop()

# =============================
# 2. Tabs untuk Input
# =============================
tab_camera, tab_upload = st.tabs(["📸 Ambil Foto dari Kamera", "⬆️ Unggah File Foto"])

img = None
input_source = None # Untuk melacak sumber input

with tab_camera:
    st.subheader("Ambil Foto Wajah")
    camera_image = st.camera_input("Klik untuk mengambil foto")

    if camera_image:
        # Convert gambar kamera → numpy array
        file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_source = "Kamera"
        
with tab_upload:
    st.subheader("Unggah Foto Wajah")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Konversi UploadedFile Streamlit ke format yang bisa dibaca cv2
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_source = "Unggahan"

# =============================
# 3. Proses dan Prediksi
# =============================

if img is not None:
    st.markdown("---")
    st.subheader("Gambar Input")
    # Tampilkan gambar input (channels="BGR" memastikan warna ditampilkan dengan benar oleh Streamlit/OpenCV)
    st.image(img, caption=f"Gambar dari {input_source}", channels="BGR", use_column_width=True)

    # Prediksi Model
    result = predict_identity(model, img)

    st.markdown("## 🧠 Hasil Analisis Model")

    # Hasil utama
    st.success(f"**Identitas yang Diprediksi:** {result['predicted_name']}")
    st.write(f"**Confidence:** {result['confidence']*100:.2f}%")

    # Top 5 ranking
    st.markdown("### 🔍 Top-5 Prediksi Terdekat:")
    for item in result["top5"]:
        st.write(f"- **{item['name']}** — {item['confidence']*100:.2f}%")