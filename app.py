#Import requied libraries
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models # type: ignore
import numpy as np

# Load pretrained ResNet model
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Convert image using PIL
def get_image_embedding(image_path):
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(image).flatten().numpy()
    return embedding

def compare_casting_numbers_resnet(img1_path, img2_path, threshold=0.95):
    emb1 = get_image_embedding(img1_path)
    emb2 = get_image_embedding(img2_path)
    
    # Cosine similarity calculation
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if similarity > threshold:
        print(f"TRUE Casting numbers match! (Similarity: {similarity:.2f})")
    else:
        print(f" FALSE Casting numbers DO NOT match! (Similarity: {similarity:.2f})")

# Image comparision
image1_path = "C:\\Casting number detection\\Casting metal.jpeg"
image2_path = "C:\\Casting number detection\\casting metal 3.jpg"
compare_casting_numbers_resnet(image1_path, image2_path)
