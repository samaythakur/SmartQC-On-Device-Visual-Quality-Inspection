"""
SmartQC — proper evaluation metrics.
 
Loads the trained weights, re-creates the exact same validation split
used during training (same seed), and reports:
  - Confusion matrix
  - Precision / Recall / F1 per class
  - ROC curve + AUC
 
Saves two plots: model/eval_confusion_matrix.png, model/eval_roc_curve.png
 
Run (from the SmartQC project root, with venv active):
    pip install scikit-learn matplotlib
    python eval_metrics.py
"""
 
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
 
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, classification_report,
)
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------------------
# Config — must match train.py exactly, so we evaluate on the SAME
# validation split the model was validated on during training.
# ---------------------------------------------------------------------------
DATA_DIR = Path("Dataset-Fresh & Mold Bread Images")
WEIGHTS_PATH = Path("model/smartqc_weights.pt")
OUTPUT_DIR = Path("model")
 
IMG_SIZE = 224
BATCH_SIZE = 16
VAL_SPLIT = 0.2
SEED = 42
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def main():
    torch.manual_seed(SEED)
 
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=eval_tf)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
 
    # Reproduce the exact same split as train.py (same seed, same order of ops)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    _, val_ds = random_split(full_dataset, [train_size, val_size])
 
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Validation set size: {len(val_ds)}")
 
    # -- Load model ------------------------------------------------------
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(DEVICE)
    model.eval()
 
    # -- Run inference on validation set ----------------------------------
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
 
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class index 1
 
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
 
    # -- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
 
    # -- Precision / Recall / F1 -------------------------------------------
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name}: precision={precision[i]:.3f}  recall={recall[i]:.3f}  f1={f1[i]:.3f}  support={support[i]}")
 
    print("\nFull classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
 
    # -- ROC curve + AUC ----------------------------------------------------
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC-AUC: {roc_auc:.4f}")
 
    # -- Plot confusion matrix ------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short_names = ["Good" if "good" in c.lower() or "fresh" in c.lower() else "Defective" for c in class_names]
    ax.set_xticklabels(short_names)
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "eval_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    print(f"\nSaved confusion matrix plot to: {cm_path.resolve()}")
 
    # -- Plot ROC curve -------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    ax2.plot(fpr, tpr, color="#065A82", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    roc_path = OUTPUT_DIR / "eval_roc_curve.png"
    fig2.savefig(roc_path, dpi=150)
    print(f"Saved ROC curve plot to: {roc_path.resolve()}")
 
 
if __name__ == "__main__":
    main()
 