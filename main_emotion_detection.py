# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:06:10 2025

@author: haris_tuytv90
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections, BoxAnnotator
from PIL import Image

# Paths for input/output folders
frame_output_folder = r"/content/drive/MyDrive/emotion_detection/extracted_frames"  # Folder containing extracted frames
output_folder = r"/content/drive/MyDrive/emotion_detection/output_faces"  # Folder to save processed results
crops_folder = os.path.join(output_folder, "/content/drive/MyDrive/emotion_detection/cropped_faces")
enhanced_folder = os.path.join(output_folder, "/content/drive/MyDrive/emotion_detection/enhanced_faces")

# Create directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(crops_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)

# Download and load YOLOv8 face detection model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# Initialize box annotator for drawing bounding boxes
box_annotator = BoxAnnotator(thickness=2)

# Function to check if an image is blurry (used for small faces)
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Function to apply super-resolution (basic upscaling using OpenCV)
def upscale_image(image, scale=2):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

# Process all frames in the folder
for image_file in os.listdir(frame_output_folder):
    image_path = os.path.join(frame_output_folder, image_file)

    # Read the image using PIL and convert to numpy array for OpenCV
    pil_image = Image.open(image_path)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run YOLO face detection model
    output = model(image)
    detections = Detections.from_ultralytics(output[0])

    # Save the image with bounding boxes drawn
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image_path = os.path.join(output_folder, f"annotated_{image_file}")
    cv2.imwrite(annotated_image_path, annotated_image)

    print(f"Processed {image_file} - Found {len(detections)} faces")

    # Process detected faces: crop, filter, and upscale
    if len(detections) > 0:
        boxes = detections.xyxy  # Get bounding boxes (xyxy format)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
            cropped_face = image[y1:y2, x1:x2].copy()  # Crop face

            # Filter out faces that are smaller than 50x50
            if cropped_face.shape[0] < 50 or cropped_face.shape[1] < 50:
                continue  # Skip small faces

            # Save the cropped face
            crop_filename = f"{os.path.splitext(image_file)[0]}_face{i+1}.jpg"
            crop_path = os.path.join(crops_folder, crop_filename)
            cv2.imwrite(crop_path, cropped_face)

            # Apply super-resolution (upscaling) to the cropped face
            enhanced_face = upscale_image(cropped_face, scale=2)

            # Save the enhanced (upscaled) face
            enhanced_filename = f"{os.path.splitext(image_file)[0]}_enhanced_face{i+1}.jpg"
            enhanced_path = os.path.join(enhanced_folder, enhanced_filename)
            cv2.imwrite(enhanced_path, enhanced_face)

            print(f"Saved enhanced face: {enhanced_filename}")

print("Processing complete!")