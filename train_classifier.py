import os
import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = "data"  # Path to the root directory containing unsplit data
SPLIT_DIR = "split_data"  # Directory to store split data
MODEL_SAVE_DIR = "saved_models"  # Directory to save trained models
REPORTS_SAVE_DIR = "reports"  # Directory to save classification reports
BATCH_SIZE = 32
NUM_CLASSES = 5  # Number of instrument categories
TEST_SPLIT = 0.2  # Fraction of data to use for testing
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(REPORTS_SAVE_DIR, exist_ok=True)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Transform input size for all three models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Retrieve class names
class_names = train_dataset.classes

# Define transfer learning function
def create_model(model_name):
    if model_name == "resnet18": # Transfer learning using ResNet-18
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "efficientnet_b0": # Transfer learning using EfficientNet-B0
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )
    elif model_name == "vgg16": # Transfer learning using VGG-16
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(DEVICE)

# Training and evaluation functions
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses, test_accuracies = [], []

    for epoch in range(epochs): # Iterate over epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Optimize the model
            
            running_loss += loss.item() 
        
        train_loss = running_loss / len(train_loader) # Calculate loss over the entire training set
        train_losses.append(train_loss)

        test_accuracy = evaluate_model(model, test_loader) # Evaluate the model on the test set
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    return train_losses, test_accuracies

# Function to evaluate model performance on the test set
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

# Function to plot training loss and test accuracy
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
    plt.close()  # Close the figure to free memory
    # plt.show()

# Function to evaluate detailed metrics such as classification report and confusion matrix
def evaluate_detailed_metrics(model, test_loader, class_names, model_name):
    """
    Computes and saves the classification report and confusion matrix for the given model.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_path = os.path.join(REPORTS_SAVE_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for {model_name}:\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    cm_filename = f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()  # Close the figure to free memory
    print(f"Confusion matrix saved to {cm_filename}")

# Train and evaluate all models
models_to_train = ["resnet18", "efficientnet_b0", "vgg16"]
# models_to_train = ["vgg16"]
all_train_losses = {}
all_test_accuracies = {}

# Iterate over each model
for model_name in models_to_train:
    print(f"\nTraining {model_name}...")
    model = create_model(model_name) # Initialize the model
    criterion = nn.CrossEntropyLoss() # Set the loss function to cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Set the optimizer to Adam

    train_losses, test_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS)
    all_train_losses[model_name] = train_losses
    all_test_accuracies[model_name] = test_accuracies

    # Save the trained model
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot individual results
    plot_results({model_name: train_losses}, "Training Loss", f"{model_name}_loss.png")
    plot_results({model_name: test_accuracies}, "Test Accuracy", f"{model_name}_accuracy.png")
    print(f"Training loss and test accuracy plots saved for {model_name}")

    # Evaluate additional metrics
    evaluate_detailed_metrics(model, test_loader, class_names, model_name)

# Plot combined results
plot_results(all_train_losses, "Training Loss", "combined_loss.png")
plot_results(all_test_accuracies, "Test Accuracy", "combined_accuracy.png")
print("Combined training loss and test accuracy plots saved.")