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
SPLIT_DIR = "split_data"  # Directory to store split data
PLOT_FILE = "training_plot.png"  # File to save the training plot
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

# Load pre-trained ResNet and modify it
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(DEVICE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        # Training phase
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

        # Evaluation phase
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return train_losses, test_accuracies

# Evaluation loop
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

# Plot training loss and accuracy
def plot_training_results(train_losses, test_accuracies, plot_file):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.title("Training Loss and Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.show()

# Train and evaluate the model
if __name__ == "__main__":
    print("Starting training...")
    train_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS)
    print("Evaluating model...")
    evaluate_model(model, test_loader)
    print(f"Saving training plot to {PLOT_FILE}...")
    plot_training_results(train_losses, test_accuracies, PLOT_FILE)
