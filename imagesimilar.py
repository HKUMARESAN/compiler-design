import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Pretrained ResNet50 (no warnings)
# -------------------------
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])  # remove last classification layer
model.eval()

# Preprocessing (resize + normalize for ResNet50)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_embedding(image_path):
    """Extract 2048-dim embedding for one image"""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img).squeeze().numpy()
    return embedding

# -------------------------
# 2. Paths (update here if needed)
# -------------------------
dataset_path = r"E:\project\dataset"   # your dataset folder
query_path = r"E:\project\234.jpg"     # your query image

# -------------------------
# 3. Build database embeddings
# -------------------------
database = {}
for file in os.listdir(dataset_path):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(dataset_path, file)
        database[file] = get_embedding(path)

# -------------------------
# 4. Query image embedding
# -------------------------
query_emb = get_embedding(query_path)

# -------------------------
# 5. Similarity search (Top-5)
# -------------------------
scores = []
for fname, emb in database.items():
    sim = cosine_similarity([query_emb], [emb])[0][0]
    scores.append((fname, sim))

# Sort by similarity (descending)
scores.sort(key=lambda x: x[1], reverse=True)

# Pick Top-5
top5 = scores[:5]

print("Query image:", query_path)
print("Top 5 most similar images:")
for fname, sim in top5:
    print(f"   {fname}  (similarity = {sim:.4f})")

# -------------------------
# 6. Show Query + Top-5 Matches
# -------------------------
query_img = Image.open(query_path)

plt.figure(figsize=(15, 6))

# Show query first
plt.subplot(1, 6, 1)
plt.imshow(query_img)
plt.title("Query")
plt.axis("off")

# Show top-5 matches
for i, (fname, sim) in enumerate(top5, start=2):
    img = Image.open(os.path.join(dataset_path, fname))
    plt.subplot(1, 6, i)
    plt.imshow(img)
    plt.title(f"Sim {sim:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()
