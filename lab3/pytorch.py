import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Define data augmentation transformations
data_augmentation_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01)
])

# MNIST dataset with data augmentation
class MNISTAugmentedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

# Load the MNIST dataset
train_dataset = datasets.MNIST('data/', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST('data/', train=False, download=True, transform=ToTensor())

# Split train_dataset into train and validation sets
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Apply data augmentation to train_dataset
train_dataset_augmented = MNISTAugmentedDataset(train_dataset, transform=data_augmentation_transform)

# DataLoader
train_loader = DataLoader(train_dataset_augmented, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Plot original and augmented images with labels
def plot_images(original_images, augmented_images, dataset, num_images=3, save_file="images_plot.png"):
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # Original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i][0].squeeze(), cmap='gray')
        plt.title(f"Original: {dataset[i][1]}")
        plt.axis("off")

        # Augmented image
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(augmented_images[i][0].squeeze(), cmap='gray')
        plt.title("Augmented ")
        plt.axis("off")
    plt.savefig(save_file)
    plt.show()

# Plot predictions of the best model
def plot_predictions(model, test_loader, num_images=3, top_n=3, save_file=None):
    plt.figure(figsize=(15, 6))
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            # Select randomly a batch from the test_loader
            images, labels = next(iter(test_loader))
            outputs = model(images)
            predicted_classes = torch.argsort(outputs, descending=True, dim=1)[:, :top_n]
            predicted_probabilities = torch.softmax(outputs, dim=1)

            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')

            # Get top classes and their probabilities
            top_classes = predicted_classes[i]
            top_probs = predicted_probabilities[i][top_classes]

            # Format title text
            title_text = '\n'.join([f'Class: {cls.item()}, Probability: {prob.item():.2%}' for cls, prob in zip(top_classes, top_probs)])

            plt.title(title_text, color='#017653')
            plt.axis("off")
    if save_file:
        plt.savefig(save_file)
    plt.show()

# Initialize the model, optimizer, and loss function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
epochs = 10  # Change the number of epochs as needed
best_val_loss = float('inf')
best_accuracy = 0.0
last_val_loss = float('inf')
last_accuracy = 0.0
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', total=len(train_loader))
    for images, labels in pbar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'Loss': loss.item()}, refresh=False)
    pbar.close()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    accuracy = correct / total

    # Checkpointing
    if val_loss < best_val_loss:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': val_loss,
            'best_accuracy': accuracy,
        }, 'best_model_checkpoint.pth')
        best_val_loss = val_loss
        best_accuracy = accuracy

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_val_loss': val_loss,
        'last_accuracy': accuracy,
    }, 'last_model_checkpoint.pth')
    last_val_loss = val_loss
    last_accuracy = accuracy

# Load the best model and metrics
checkpoint = torch.load('best_model_checkpoint.pth')
best_model = CNN()
best_model.load_state_dict(checkpoint['model_state_dict'])
best_optimizer = optim.Adam(best_model.parameters(), lr=0.001)
best_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
best_val_loss = checkpoint['best_val_loss']
best_accuracy = checkpoint['best_accuracy']

# Load the last model and metrics
checkpoint = torch.load('last_model_checkpoint.pth')
last_model = CNN()
last_model.load_state_dict(checkpoint['model_state_dict'])
last_optimizer = optim.Adam(last_model.parameters(), lr=0.001)
last_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
last_val_loss = checkpoint['last_val_loss']
last_accuracy = checkpoint['last_accuracy']

# Evaluate the trained model and best model
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
    accuracy = correct / total
    loss /= len(test_loader)
    return loss, accuracy

# Evaluate the last model
last_model_loss, last_model_accuracy = evaluate_model(last_model, test_loader)

# Evaluate the best model
best_model_loss, best_model_accuracy = evaluate_model(best_model, test_loader)

# Print test loss and accuracy for both models
print("Last Model:")
print("Test Loss:", last_model_loss)
print("Test Accuracy:", last_model_accuracy)
print("\nBest Model:")
print("Test Loss:", best_model_loss)
print("Test Accuracy:", best_model_accuracy)

# Write model metrics to a text file
best_val_loss_str = f'Best Validation Loss: {best_val_loss:.4f}'
best_accuracy_str = f'Best Accuracy: {best_accuracy:.4f}'
last_val_loss_str = f'Last Validation Loss: {last_val_loss:.4f}'
last_accuracy_str = f'Last Accuracy: {last_accuracy:.4f}'

with open('model_metrics.txt', 'w') as file:
    file.write(best_val_loss_str + '\n')
    file.write(best_accuracy_str + '\n')
    file.write(last_val_loss_str + '\n')
    file.write(last_accuracy_str + '\n')

# Plot and save images plotted before
plot_images(train_dataset, train_dataset_augmented, train_dataset, save_file="augmented_images_plot.png")
plot_predictions(best_model, test_loader, save_file="best_model_predictions.png")
plot_predictions(last_model, test_loader, save_file="last_model_predictions.png")
