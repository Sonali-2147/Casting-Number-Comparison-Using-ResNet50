# Import required libraries
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pytesseract
import cv2

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

# Image Preprocessing 
# Image cleaning
def preprocess_image_for_ocr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # Invert the image for better contrast
    inverted = cv2.bitwise_not(thresh)
    return Image.fromarray(inverted)

# Converting image using PIL and extract features
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(image).flatten().numpy()
    print(f"\nExtracted Features from {image_path}:", embedding)
    
    # Extracting numbers using Tesseract OCR with preprocessing
    preprocessed_image = preprocess_image_for_ocr(image_path)
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    print(f"Raw OCR Output: {text}")
    numbers = ''.join(filter(str.isdigit, text))
    print(f"Extracted Numbers from {image_path}: {numbers}\n")
    return embedding, numbers

# Compare casting numbers using extracted features
def compare_casting_numbers_resnet(img1_path, img2_path, threshold=0.95):
    print(f"\nComparing images:\nImage 1: {img1_path}\nImage 2: {img2_path}\n")
    emb1, numbers1 = get_image_embedding(img1_path)
    emb2, numbers2 = get_image_embedding(img2_path)
    
    # Display both extracted numbers
    print(f"Extracted Summary:\nImage 1: {numbers1}\nImage 2: {numbers2}\n")
    
    # Cosine similarity calculation
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Cosine Similarity: {similarity:.2f}")
    if similarity > threshold and numbers1 == numbers2:
        print(f"TRUE: Casting numbers match! (Similarity: {similarity:.2f})")
    else:
        print(f"FALSE: Casting numbers DO NOT match! (Similarity: {similarity:.2f})")

# Image comparison
image1_path = "C:\\Users\\kadam\\Downloads\\Casting-Number-Comparison-Using-ResNet50-main\\Casting-Number-Comparison-Using-ResNet50-main\\casting metal 3.jpg"
image2_path = "C:\\Users\\kadam\\Downloads\\Casting-Number-Comparison-Using-ResNet50-main\\Casting-Number-Comparison-Using-ResNet50-main\\Casting metal.jpeg"
compare_casting_numbers_resnet(image1_path, image2_path)
