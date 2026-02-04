import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os

# Config
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Datasets and Loaders
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

class_names = train_ds.classes
num_classes = len(class_names)

# Model
model = models.resnet50(pretrained=True)
    # Freeze backbone
for param in model.parameters():
    param.requires_grad = False
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

# Loss Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.items()

    avg_loss = running_loss/len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "city_skyline_model.pth")
print("Model Saved!")

# Single Image Prediction
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        idx = probs.argmax(1).item()

    return class_names[idx], probs[0][idx].item()