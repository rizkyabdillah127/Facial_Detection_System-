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
    model_path = "model/model_efficientnet.pth"
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
        # Buat model dummy untuk mencegah error total jika model tidak ditemukan
        return None
        
    return model

# ===========================
# 2. Mapping Label → Nama
# ===========================
# 🔴 PERHATIAN: PASTIKAN PATH INI MENGARAH KE FOLDER UTAMA DATASET (Folder yang berisi 70 sub-folder nama)
DATASET_ROOT_PATH = r"C:\Dyo\Kuliah\Deep Learning\Facial_Detection_System-\Mahasiswa"

def get_label_map(dataset_root):
    """Memuat pemetaan indeks kelas ke nama dari struktur folder dataset."""
    if not os.path.isdir(dataset_root):
        print(f"ERROR: Folder dataset tidak ditemukan di: {dataset_root}")
        print("Menggunakan label generik (Person 1, Person 2, ...)")
        return {i: f"Person {i+1}" for i in range(70)}
        
    try:
        # ImageFolder akan memindai folder dan membuat pemetaan class_to_idx.
        image_dataset = datasets.ImageFolder(root=dataset_root)
        
        # Mapping ImageFolder adalah {'Nama': Indeks}, kita balik menjadi {Indeks: 'Nama'}
        class_to_idx = image_dataset.class_to_idx
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        
        print("Pemetaan kelas berhasil dimuat dari folder dataset.")
        return idx_to_class

    except Exception as e:
        print(f"ERROR: Gagal memuat struktur dataset dari path: {dataset_root}")
        print(f"Detail error: {e}")
        print("Menggunakan label generik (Person 1, Person 2, ...)")
        # Fallback jika terjadi kesalahan
        return {i: f"Person {i+1}" for i in range(70)}

# Panggil fungsi untuk mendapatkan label_map yang akurat
label_map = get_label_map(DATASET_ROOT_PATH)


# ===========================
# 3. Analisis Foto + Laporan
# ===========================
def predict_identity(model, img):
    # Cek apakah model berhasil dimuat (diperlukan jika load_model mengembalikan None)
    if model is None:
        return {
            "predicted_class": -1,
            "predicted_name": "Error: Model tidak dimuat",
            "confidence": 0.0,
            "top5": []
        }

    # Konversi gambar jika memiliki channel alfa (RGBA)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img_resized = cv2.resize(img, (224, 224)) / 255.0
    # Konversi ke Tensor PyTorch (HWC -> CHW) dan tambahkan batch dimension (1, C, H, W)
    img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.softmax(logits, dim=1)[0]

        pred_class = torch.argmax(prob).item()
        # Menggunakan label_map yang baru dimuat
        pred_name = label_map.get(pred_class, f"Person {pred_class+1} (Unknown)") 
        confidence = prob[pred_class].item()

        top5_val, top5_idx = torch.topk(prob, 5)
        top5 = [
            {
                "class_id": idx.item(),
                # Menggunakan label_map yang baru dimuat
                "name": label_map.get(idx.item(), f"Person {idx.item()+1} (Unknown)"), 
                "confidence": val.item()
            } for idx, val in zip(top5_idx, top5_val)
        ]

    return {
        "predicted_class": pred_class,
        "predicted_name": pred_name,
        "confidence": confidence,
        "top5": top5
    }