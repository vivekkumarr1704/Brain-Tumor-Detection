import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
train_dir = "dataset/train"
val_dir = "dataset/val"
num_classes = 2
batch_size = 16
epochs = 5  # fast training

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# -----------------------------
# SSD-MOBILE NET CLASSIFIER
# -----------------------------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Replace classifier head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# TRAINING LOOP
# -----------------------------
print("\nTraining SSD-MobileNet classifier...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    acc = val_correct / val_total * 100
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.3f} | Val Acc: {acc:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "models/ssd/ssd_mobilenet_classifier.pth")

print("\nTraining complete! Model saved as ssd_mobilenet_classifier.pth")
