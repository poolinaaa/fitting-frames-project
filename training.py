import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"On device : {device}")

drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/face_shape_dataset ./

TRAIN_PATH = "./face_shape_dataset/fs/training_set"
TEST_PATH = "./face_shape_dataset/fs/testing_set"

example_image_path = random.choice(os.listdir(os.path.join(TRAIN_PATH, 'Oval')))
example_image_path = os.path.join(TRAIN_PATH, 'Oval', example_image_path)
example_image = Image.open(example_image_path)

transformations = [
    ("Original", None),
    ("Resize", T.Resize((224, 224))),
    ("RandomHorizontalFlip", T.RandomHorizontalFlip()),
    ("RandomRotation", T.RandomRotation(degrees=10)),
    ("ColorJitter", T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)),
]

fig, axes = plt.subplots(1, len(transformations), figsize=(20, 5))

for i, (title, transform) in enumerate(transformations):
    ax = axes[i]
    ax.axis('off')

    if transform:
        transformed_image = transform(example_image)
    else:
        transformed_image = example_image

    ax.imshow(transformed_image)
    ax.set_title(title)

plt.show()

train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() - 1

def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, SyntaxError):
        return Image.new('RGB', (224, 224))

train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=train_transforms, loader=safe_pil_loader)
test_dataset = datasets.ImageFolder(root=TEST_PATH, transform=test_transforms, loader=safe_pil_loader)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(train_dataset.class_to_idx)

torch.manual_seed(42)

model = torchvision.models.efficientnet_b4(pretrained=True)
num_classes = len(train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999), weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

model.to(device)
num_epochs = 25
best_val_loss = 1_000_000

scaler = GradScaler()

train_losses_history = []
val_losses_history = []
accuracies_history = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_losses_history.append(avg_train_loss)

    model.eval()
    validation_losses = []
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_val_loss = sum(validation_losses) / len(validation_losses)
        val_losses_history.append(avg_val_loss)

        accuracy = accuracy_score(all_labels, all_predictions)
        accuracies_history.append(accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = '/content/drive/MyDrive/my_model/best_model.pth'
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

print(f'Best Validation Loss: {best_val_loss:.4f}')


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(range(1, num_epochs + 1), train_losses_history, label='Train Loss', color='blue')
axes[0].plot(range(1, num_epochs + 1), val_losses_history, label='Validation Loss', color='red')
axes[0].set_title('Training and Validation Loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(range(1, num_epochs + 1), accuracies_history, label='Accuracy', color='green')
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.show()

df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_losses_history,
    'Validation Loss': val_losses_history,
    'Accuracy': accuracies_history
})
df.to_csv('/content/drive/MyDrive/my_model/training_results.csv', index=False)

model_save_path = '/content/drive/MyDrive/my_model/model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)

print(f'Final model saved at {model_save_path}')

with open('/content/drive/MyDrive/my_model/training_summary.txt', 'w') as f:
    f.write(f"Final Accuracy: {accuracies_history[-1]:.4f}\n")
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")

model = torchvision.models.efficientnet_b4(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

model.load_state_dict(torch.load(model_save_path))
model.to(device)

model.eval()
all_predictions = []
all_labels = []

with torch.inference_mode():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Final accuracy on test set: {accuracy:.4f}")
