import torch
import cv2
import os
import numpy as np
from torchvision.models import efficientnet_b0
import torchvision.datasets as datasets 

# ===========================
# 1. Load Model EfficientNet-B0
# ===========================
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "model",
        "model_efficientnet.pth"
    )
    model_path = os.path.abspath(model_path)

    model = efficientnet_b0(weights=None)

    # Output layer untuk 70 kelas
    model.classifier[1] = torch.nn.Linear(1280, 70)

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print("Model EfficientNet-B0 (70 kelas) berhasil dimuat.")
    except Exception as e:
        print(f"ERROR: Gagal memuat model dari {model_path}. Detail: {e}")
        return None
        
    return model

# ===========================
# 2. Mapping Label → Nama
# ===========================

# Mengambil lokasi file ini
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Root project (naik 1 folder)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Folder dataset Mahasiswa (otomatis)
DATASET_ROOT_PATH = os.path.join(PROJECT_ROOT, "Mahasiswa")
DATASET_ROOT_PATH = os.path.abspath(DATASET_ROOT_PATH)

def get_label_map(dataset_root):
    """Memuat pemetaan indeks kelas ke nama berdasarkan struktur folder dataset."""
    if not os.path.isdir(dataset_root):
        print(f"ERROR: Folder dataset tidak ditemukan di: {dataset_root}")
        print("Menggunakan label generik (Person 1, Person 2, ...)")
        return {i: f"Person {i+1}" for i in range(70)}
        
    try:
        # ImageFolder mendeteksi kelas berdasarkan nama folder
        image_dataset = datasets.ImageFolder(root=dataset_root)
        
        class_to_idx = image_dataset.class_to_idx
