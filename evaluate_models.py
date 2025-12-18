import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG --------
TEST_DIR = "dataset/test"
YOLO_PATH = "runs/classify/train2/weights/best.pt"
FASTER_PATH = "models/faster_rcnn/faster_rcnn_classifier.pth"
RETINA_PATH = "models/retinanet/retinanet_classifier.pth"
SSD_PATH = "models/ssd/ssd_mobilenet_classifier.pth"
MASK_PATH = "models/mask_rcnn/maskrcnn_classifier.pth"

OUT_DIR = "results/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", DEVICE)

# -------- LOADERS --------

def load_image(path):
    return Image.open(path).convert("RGB")

def save_confusion(cm, classes, fname):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

def load_resnet(path):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return model, tf

print("Loading models...")

faster_model, faster_tf = load_resnet(FASTER_PATH)
retina_model, retina_tf = load_resnet(RETINA_PATH)
mask_model, mask_tf = load_resnet(MASK_PATH)

ssd_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
ssd_model.classifier[1] = nn.Linear(ssd_model.classifier[1].in_features, 2)
ssd_state = torch.load(SSD_PATH, map_location="cpu")
ssd_model.load_state_dict(ssd_state)
ssd_model.to(DEVICE)
ssd_model.eval()
ssd_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- TEST IMAGES --------

classes = sorted([d.name for d in Path(TEST_DIR).iterdir() if d.is_dir()])
print("Classes:", classes)

test_files = []
for cls in classes:
    for ext in ("*.jpg","*.jpeg","*.png"):
        for fp in (Path(TEST_DIR)/cls).glob(ext):
            test_files.append((str(fp), cls))

print("Total test images:", len(test_files))

def predict_yolo(img):
    res = yolo(img)
    probs = res[0].probs.data.tolist()
    names = res[0].names
    idx = int(np.argmax(probs))
    return names[idx], float(probs[idx])

def predict_resnet(model, tf, img):
    t = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        p = torch.softmax(out,1).cpu().numpy()[0]
    idx = int(np.argmax(p))
    label = "no_tumor" if idx == 0 else "tumor"
    return label, float(p[idx])

# -------- MODEL LIST --------

models_info = {
    "YOLOv8": lambda img: predict_yolo(img),
    "FasterRCNN": lambda img: predict_resnet(faster_model, faster_tf, img),
    "MaskRCNN": lambda img: predict_resnet(mask_model, mask_tf, img),
    "RetinaNet": lambda img: predict_resnet(retina_model, retina_tf, img),
    "SSD-MobileNet": lambda img: predict_resnet(ssd_model, ssd_tf, img)
}

summary_rows = []
all_rows = []

for name, fn in models_info.items():
    print("\nEvaluating:", name)

    preds = []
    trues = []

    rows = []

    for fp, true_label in test_files:
        img = load_image(fp)
        pred_label, conf = fn(img)

        preds.append(pred_label)
        trues.append(true_label)

        rows.append({
            "model": name,
            "file": fp,
            "true": true_label,
            "pred": pred_label,
            "conf": conf
        })

    # METRICS
    label_idx = {"no_tumor":0, "tumor":1}
    y_true = [label_idx[t] for t in trues]
    y_pred = [label_idx[p] for p in preds]

    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    prec = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
    rec = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
    f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)


    cm = confusion_matrix(y_true, y_pred)
    save_confusion(cm, classes, f"{OUT_DIR}/cm_{name}.png")

    summary_rows.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": f"{OUT_DIR}/cm_{name}.png"
    })

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/{name}_results.csv", index=False)

    all_rows.extend(rows)

pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/metrics_summary.csv", index=False)
pd.DataFrame(all_rows).to_csv(f"{OUT_DIR}/all_results.csv", index=False)

print("\nEvaluation complete!")
print("Summary saved to results/metrics/")
