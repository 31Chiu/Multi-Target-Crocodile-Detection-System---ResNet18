import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets                    # Use our own customized dataset module
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm                               # To show a progress bar

# Step 1: Definitions and Preparations
def get_device():
    # Gets the available compute device (GPU or CPU).
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pretrained_resnet18(model_path='best_resnet18_model.pth', num_classes=10):
    """
    Loads a pre-trained ResNet18 model and modifies it into a feature extractor.
    We remove the final classification layer, so the model's output is the feature vector.
    """
    # Load the model structure
    model = models.resnet18(weights=None)   # weights=None as we will load our own

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    # Replace the final layer with one that matches your number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the weights you saved from your training script
    model.load_state_dict(torch.load(model_path, map_location=get_device()))

    # Remove the final fully-connected layer to make it a feature extractor
    # We use nn.Identity() as a placeholder that does nothing
    model.fc = nn.Identity()

    device = get_device()
    model = model.to(device)
    model.eval()    # Set the model to evaluation mode
    return model

# Step 2: Feature Extraction
def extract_features(dataloader, model):
    # Iterates through the dataset and use the ResNet18 model to extract features from all images.
    features_list = []
    labels_list = []
    device = get_device()

    with torch.no_grad():   # No need to calculate gradients, saves memory and computation
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            # The model's output is now a batch of feature vectors
            feature_batch = model(images)

            # Move features and labels to CPU and convert to NumPy arrays
            features_list.append(feature_batch.cpu().numpy())
            labels_list.append(labels.numpy())

        # Concatenate lists of arrays into single large NumPy arrays
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return features, labels
    
# Step 3: Main Execution Flow
def main():
    # Main function to orchestrate all steps.
    # !! IMPORTANT !!
    # Define the path to your dataset and the number of classes you have.
    # The dataset should be structured.
    DATASET_PATH = './datasets'
    NUM_CLASSES = 2
    MODEL_PATH = './resnet18_checkpoint/best_resnet18_model.pth'

    # 1. Load the model
    print("Loading ResNet18 feature extractor model...")
    resnet_feature_extractor = load_pretrained_resnet18(model_path=MODEL_PATH, num_classes=NUM_CLASSES)

    # 2. Prepare the dataset
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading custom dataset...")
    # Load the train and validate sets using ImageFolder
    train_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/test', transform=transform)

    # Create DataLoaders
    # Adjust batch_size based on your hardware's capability
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Extract features
    print("Extracting features from the training set...")
    train_features, train_labels = extract_features(train_loader, resnet_feature_extractor)

    print("Extracting feature from the test set...")
    test_features, test_labels = extract_features(test_loader, resnet_feature_extractor)

    print(f'Feature extraction complete! Train features shape: {train_features.shape}, Test features shape: {test_features.shape}')

    # 4. Train and evaluate the KNN classifier
    print("\n--- Training and Evaluating KNN Classifier ---")

    # Define a value for K, a key parameter for the KNN algorithm
    k_value = 5
    print(f'Using K = {k_value}')

    # Create an instance of the KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k_value, n_jobs=-1) # n_jobs=-1 uses all available CPU cores

    print("Training the KNN classifier...")
    # "Training" for KNN is simply memorizing the training data
    knn.fit(train_features, train_labels)

    print("Making predictions with the KNN classifier...")
    # Make predictions on the test set
    predictions = knn.predict(test_features)

    # 5. Calculate and display the accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print("------------------------------------------")
    print(f'KNN classifier accuracy on the test set: {accuracy * 100:.2f}%')
    print("------------------------------------------")

if __name__ == '__main__':
    main()
