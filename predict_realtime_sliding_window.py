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
# Time for measuring frame performance
import time
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

def process_patch(patch):
    """
    Process a video patch for model input
    Args:
        patch: BGR format patch from OpenCV
    Returns:
        torch.Tensor: Processed image tensor ready for model input
    """
    # Convert BGR patch to RGB and create PIL Image object
    img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

    # Define image transformations pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),         # Resize image to 224x224
        transforms.ToTensor(),          # Convert to tensor and scale to [0,1]
        transforms.Normalize(           # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply transformations and add batch dimension
    return transform(img).unsqueeze(0)

# Sliding window generator
def sliding_window(image, step_size, window_size):
    # Generate sliding windows over the input image
    # Slide the window from top to bottom and from left to right
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Yield the window coordinates and the window itself
            # Generate the coordinates and image content of the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def main():
    """
    Main function to run real-time object detection
    """
    # Path to the trained model checkpoint
    model_path = 'resnet18_checkpoint/best_resnet18.pth'
    print("Loading model...")
    # Load the model and get class names
    model, class_names = load_model(model_path)
    TARGET_CLASS = 'crocodile'
    if TARGET_CLASS not in class_names:
        print(f"Error: Target class '{TARGET_CLASS}' not found in class list {class_names}.")
        return
    print(f"Model loaded successfully! Can detect objects: {TARGET_CLASS}")

    # Initialize video capture from default camera (index 0)
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    print("\nPress the 'q' key to exit real-time analysis. . .")

    # Define sliding window parameters
    (winW, winH) = (128, 127)   # Window width and height
    stepSize = 32               # The pixel distance of each slide
    CONF_THRESHOLD = 0.90       # Confidence threshold for predictions

    # Main processing loop
    while True:
        # # Start time for frame processing
        start_time = time.time() 
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Sliding window detection
        # Stores all "crocodile" windows detected in the current frame
        detections = []

        # Sliding window core logic
        # Traverse all window positions
        for (x, y, window) in sliding_window(frame, step_size=stepSize, window_size=(winW, winH)):
            # If the window size does not match the expected size, skip it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Process the window patch
            # Handling small window slices
            image_tensor = process_patch(window)
            # Make predictions
            predicted_class, confidence = predict(model, image_tensor, class_names)

            # Check if the predicted class is the target class and meets the confidence threshold
            if predicted_class == TARGET_CLASS and confidence >= CONF_THRESHOLD:
                # Append the detection to the list
                # Save the coordinates of this window
                detections.append((x, y, x + winW, y + winH))

        # Draw rectangles around detected objects
        for (startX, startY, endX, endY) in detections:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, TARGET_CLASS, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate and display FPS
        end_time = time.time()  # End time for frame processing
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with predictions
        cv2.imshow("Sliding Window Detection", frame)

        # Check for 'q' key press to exit (waitKey returns -1 if no key is pressed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()           # Release the camera
    cv2.destroyAllWindows() # Close all OpenCV windows
    print("Camera closed, program ended.")

if __name__ == '__main__':
    main()
