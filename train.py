"""
SmartQC — training script
 
Fine-tunes MobileNetV2 on the Fresh/Mold bread dataset and saves the
trained weights to model/smartqc_weights.pt
 
Run (from the SmartQC project root, with venv active):
    pip install torch torchvision
    python train.py
"""
 
import copy
from pathlib import Path
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
DATA_DIR = Path("Dataset-Fresh & Mold Bread Images")
OUTPUT_DIR = Path("model")
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = OUTPUT_DIR / "smartqc_weights.pt"
 
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def main():
    torch.manual_seed(SEED)
 
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=train_tf)
    class_names = full_dataset.classes  # alphabetical: e.g. ["Fresh Bread(Good Bread) Images", "Mold Bread(Bad Bread) Images"]
    print(f"Found classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")
 
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    # validation set should use eval transforms (no augmentation)
    val_ds.dataset = datasets.ImageFolder(str(DATA_DIR), transform=eval_tf)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
 
    print(f"Train: {train_size}  Val: {val_size}")
 
    # -- Model ---------------------------------------------------------
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(DEVICE)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
 
        train_loss = running_loss / total
        train_acc = running_correct / total
 
        # -- Validation --
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / max(val_total, 1)
 
        print(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
 
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
 
    print(f"\nBest validation accuracy: {best_acc:.3f}")
 
    model.load_state_dict(best_weights)
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }, WEIGHTS_PATH)
    print(f"Saved weights to: {WEIGHTS_PATH.resolve()}")
 
 
if __name__ == "__main__":
    main()
 