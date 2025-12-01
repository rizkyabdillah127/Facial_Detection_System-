import torch

path = "model/model_efficientnet.pth"
data = torch.load(path, map_location="cpu")
print("TIPE:", type(data))
