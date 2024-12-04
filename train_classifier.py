import os
import shutil
import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "data"  # Path to the root directory containing unsplit data
SPLIT_DIR = "split_data"  # Directory to store split data
BATCH_SIZE = 32
NUM_CLASSES = 5  # Number of instrument categories
TEST_SPLIT = 0.2  # Fraction of data to use for testing
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define transfer learning function
def create_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(DEVICE)

# Training and evaluation functions
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return train_losses, test_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_results(results, metric, filename):
    plt.figure(figsize=(10, 6))
    for model_name, values in results.items():
        plt.plot(range(1, len(values) + 1), values, label=model_name)
    plt.title(f"Comparison of {metric} Across Models")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()

# Train and evaluate all models
models_to_train = ["resnet18", "efficientnet_b0", "vgg16"]
# models_to_train = ["vgg16"]
all_train_losses = {}
all_test_accuracies = {}

for model_name in models_to_train:
    print(f"\nTraining {model_name}...")
    model = create_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS)
    all_train_losses[model_name] = train_losses
    all_test_accuracies[model_name] = test_accuracies

    # Plot individual results
    plot_results({model_name: train_losses}, "Training Loss", f"{model_name}_loss.png")
    plot_results({model_name: test_accuracies}, "Test Accuracy", f"{model_name}_accuracy.png")

# Plot combined results
plot_results(all_train_losses, "Training Loss", "combined_loss.png")
plot_results(all_test_accuracies, "Test Accuracy", "combined_accuracy.png")
