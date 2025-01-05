# Casting-Number-Comparison-Using-ResNet50

Casting Number Comparison Using ResNet50

This Python script compares two images of casting numbers using the ResNet50 model, which is pretrained on ImageNet. The script extracts embeddings from both images and computes their cosine similarity to determine if the casting numbers match. It uses PyTorch, OpenCV, and PIL for image processing.

Requirements

Ensure you have the following libraries installed:

torch
torchvision
PIL (Pillow)
numpy
opencv-python
You can install the required dependencies via pip:

bash
Copy code
pip install torch torchvision pillow numpy opencv-python
How the Script Works
Image Loading and Transformation:

The images are loaded using PIL and transformed to match the input requirements of the ResNet50 model (size 224x224 pixels, tensor conversion, and normalization).
Embedding Extraction:

The script uses the ResNet50 model to extract embeddings from each image. The ResNet50 model is pretrained on ImageNet and we remove its last layer to get embeddings before classification.
Cosine Similarity:

After obtaining the embeddings for both images, the script calculates the cosine similarity between the two embeddings. The similarity score determines whether the casting numbers match.
Threshold:

A similarity threshold of 0.95 is set by default. If the cosine similarity exceeds this threshold, the script outputs that the casting numbers match, otherwise, it indicates they do not.
Script Usage
Step 1: Prepare Images
Ensure the images you wish to compare are in JPEG or PNG format and that they are stored in accessible file paths.

Step 2: Set File Paths
Modify the file paths in the script to point to the images you wish to compare:

python
Copy code
image1_path = "C:\\Casting number detection\\Casting metal.jpeg"
image2_path = "C:\\Casting number detection\\casting metal 3.jpg"
Step 3: Run the Script
Simply run the script, and it will print whether the casting numbers match or not based on the cosine similarity.

Example Output:
sql
Copy code
TRUE Casting numbers match! (Similarity: 0.98)
or

sql
Copy code
FALSE Casting numbers DO NOT match! (Similarity: 0.89)
Customization
Threshold: The default threshold for matching is set to 0.95. You can change this by passing a different value to the compare_casting_numbers_resnet function.
python
Copy code
compare_casting_numbers_resnet(img1_path, img2_path, threshold=0.90)
