# PyTorch deep learning framework
import torch
# Neural network modules from PyTorch
import torch.nn as nn
# Import ResNet18 model architecture
from torchvision.models import resnet18
# Python Imaging Library for image processing
from PIL import Image
# Image transformation utilities from torchvision
import torchvision.transforms as transforms
# OpenCV for real-time video capture and processing
import cv2
# NumPy for numerical operations
import numpy as np
# System-level operations
import sys
# Operating system path operations
import os

def load_model(model_path):
    """
    Load and initialize the ResNet18 model from a checkpoint file
    Args:
        model_path (str): Path to the saved model checkpoint
    Returns:
        tuple: (model, class_names) - The loaded model and list of class names
    """
    # Load the saved model checkpoint into CPU memory
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Initialize the same ResNet18 model architecture without pretrained weights
    model = resnet18(weights=None)
    # Get the number of input features for the final fully connected layer
    num_features = model.fc.in_features
    # Replace the final layer with custom classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.5),                                     # Add dropout for regularization
        nn.Linear(num_features, len(checkpoint['classes']))  # New classification layer
    )
    
    # Load the trained weights from checkpoint into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Set the model to evaluation mode (disables dropout and batch norm)
    model.eval()
    
    return model, checkpoint['classes']

def predict(model, image_tensor, class_names):
    """
    Perform prediction on an input image tensor
    Args:
        model: The loaded ResNet18 model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
    Returns:
        tuple: (predicted_class, confidence) - The predicted class name and confidence score
    """
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image_tensor)
        # Get the index of the highest score
        _, predicted = torch.max(outputs, 1)
        # Convert raw outputs to probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Return the predicted class name and its confidence score
    return class_names[predicted[0]], probabilities[0][predicted[0]].item()

def process_frame(frame):
    """
    Process a video frame for model input
    Args:
        frame: BGR format frame from OpenCV
    Returns:
        torch.Tensor: Processed image tensor ready for model input
    """
    # Convert BGR frame to RGB and create PIL Image object
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Define image transformations pipeline
    transform = transforms.Compose([
        transforms.Resize(256),         # Resize image to 256x256
        transforms.CenterCrop(224),     # Crop center to 224x224 (ResNet18 input size)
        transforms.ToTensor(),          # Convert to tensor and scale to [0,1]
        transforms.Normalize(           # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations and add batch dimension
    return transform(img).unsqueeze(0)

def main():
    """
    Main function to run real-time object detection
    """
    # Path to the trained model checkpoint
    model_path = 'resnet18_checkpoint/best_resnet18.pth'
    print("Loading model...")
    # Load the model and get class names
    model, class_names = load_model(model_path)
    print(f"Model loaded successfully! Can detect objects: {class_names}")

    # Initialize video capture from default camera (index 0)
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    print("\nPress the 'q' key to exit real-time analysis. . .")

    # Main processing loop
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Get frame dimensions (height, width, channels)
        h, w, _ = frame.shape

        # Define cropping coordinates
        top_left = (w // 2 - 112, h // 2 - 112)     # Center crop coordinates
        bottom_right = (w // 2 + 112, h // 2 + 112) # Bottom right coordinates

        box_color = (0, 0, 255)                     # Red color for the box
        box_thickness = 2                           # Thickness of the box

        # Draw the rectangle on the frame
        cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)

        # Process the frame for model input
        image_tensor = process_frame(frame)
        # Get prediction and confidence score
        predicted_class, confidence = predict(model, image_tensor, class_names)

        # Set up text display parameters
        font = cv2.FONT_HERSHEY_SIMPLEX     # Font type
        font_scale = 1                      # Font size
        color = (0, 255, 0)                 # Green color in BGR
        thickness = 2                       # Line thickness

        # Create display text with prediction results
        display_text = f"Class: {predicted_class}, Confidence: {confidence:.2%}"

        # Add text overlay to the frame
        cv2.putText(frame, display_text, (50, 50), font, font_scale, color, thickness)

        # Display the frame with predictions
        cv2.imshow("Real-time Analysis", frame)

        # Check for 'q' key press to exit (waitKey returns -1 if no key is pressed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()           # Release the camera
    cv2.destroyAllWindows() # Close all OpenCV windows
    print("Camera closed, program ended.")

if __name__ == '__main__':
    main()