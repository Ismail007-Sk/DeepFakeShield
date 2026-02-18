from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "SplitData"
MODEL_DIR = BASE_DIR / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- SETTINGS ----------
BATCH_SIZE = 8
EPOCHS = 70
LR = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- DATASETS ----------
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- MODEL ----------
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- TRAIN ----------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Accuracy: {val_acc:.2f}%")

# ---------- SAVE MODEL ----------
# Create timestamp (YYYYMMDD_HHMMSS format)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create filename with timestamp
model_path = MODEL_DIR / f"model_{timestamp}.pth"

# Save model
torch.save(model.state_dict(), model_path)

print(f"Training completed and model saved at {model_path} ✅")
