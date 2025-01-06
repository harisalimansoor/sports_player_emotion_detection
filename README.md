# sports_player_emotion_detection

## Overview
This project implements a pipeline for face detection, cropping, and enhancement from video frames or images. It utilizes the YOLOv8 model for face detection and applies preprocessing techniques such as filtering out small or blurry faces and super-resolution for face enhancement.

The goal is to extract and enhance faces from input frames for further emotion detection or other downstream tasks.

## Features
Face Detection: Uses the YOLOv8 face detection model to identify faces in images.
Face Cropping: Extracts detected faces from images.
Blurry Image Filtering: Filters out low-quality or blurry faces using a Laplacian variance threshold.
Face Enhancement: Applies basic super-resolution (upscaling) to enhance cropped faces.
Annotated Image Generation: Saves images with bounding boxes drawn around detected faces.
## Dependencies
Python 3.7+
OpenCV
NumPy
ultralytics (for YOLO model)
PIL (Pillow)
huggingface_hub
supervision
Install dependencies using pip:


pip install opencv-python numpy ultralytics pillow huggingface_hub supervision

## Files and Directories
Input Frames Folder: frame_output_folder - Directory containing the extracted frames for processing.
Output Folder: output_folder - Directory where annotated images and processed results are saved.
Cropped Faces: crops_folder - Contains cropped faces from detected bounding boxes.
Enhanced Faces: enhanced_folder - Contains upscaled/enhanced face images.
Setup and Configuration
Clone the Repository:

## How to Use
Place your input images or frames in the directory specified by frame_output_folder.

## Run the script:


python main.py
Check the output folders for results:

Annotated images: Found in output_folder.
Cropped faces: Found in crops_folder.
Enhanced faces: Found in enhanced_folder.

