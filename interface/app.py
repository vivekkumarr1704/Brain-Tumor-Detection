# ------------------------------------------------------------
# FINAL COMPLETE app.py  — WITH FULL EVALUATION SECTION
# ------------------------------------------------------------

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# CONFIG / PATHS
# -----------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

ROOT = Path.cwd()
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results" / "metrics"
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"

YOLO_PATH = "runs/classify/train2/weights/best.pt"
FASTER_PATH = "models/faster_rcnn/faster_rcnn_classifier.pth"
RETINA_PATH = "models/retinanet/retinanet_classifier.pth"
SSD_PATH = "models/ssd/ssd_mobilenet_classifier.pth"
MASK_PATH = "models/mask_rcnn/maskrcnn_classifier.pth"

# -----------------------
# UTILS
# -----------------------
def load_img(x):
    return Image.open(x).convert("RGB")

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_yolo():
    if not Path(YOLO_PATH).exists(): return None
    return YOLO(YOLO_PATH)

def load_resnet(path):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    if Path(path).exists():
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return model, tf

yolo_model = load_yolo()
faster_model, faster_tf = load_resnet(FASTER_PATH)
retina_model, retina_tf = load_resnet(RETINA_PATH)
mask_model, mask_tf = load_resnet(MASK_PATH)

ssd_model = models.mobilenet_v2(weights="IMAGENET1K_V1")
ssd_model.classifier[1] = nn.Linear(ssd_model.classifier[1].in_features, 2)
if Path(SSD_PATH).exists():
    state = torch.load(SSD_PATH, map_location="cpu")
    ssd_model.load_state_dict(state)
ssd_model.eval()
ssd_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------
# SIDEBAR
# -----------------------
page = st.sidebar.radio("Select Page", ["Prediction", "Evaluation"])

samples = ["-- none --"]
test_dir = DATASET_DIR / "test"

if test_dir.exists():
    for cls in sorted(test_dir.iterdir()):
        if cls.is_dir():
            for ext in ("*.jpg","*.png","*.jpeg"):
                for f in cls.glob(ext):
                    samples.append(str(f))

sample_choice = st.sidebar.selectbox("Choose sample image", samples)

if st.sidebar.button("Reload App"):
    st.experimental_rerun()


# ------------------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------------------
if page == "Prediction":

    st.title("Brain Tumor Detection — Multi-Model")

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Upload MRI Image", ["jpg","jpeg","png"])
        if sample_choice != "-- none --":
            uploaded = sample_choice

        if uploaded:
            img = load_img(uploaded)
            st.image(ImageOps.fit(img, (450, 450)), caption="Input MRI")
            pred_btn = st.button("Predict")
        else:
            pred_btn = False
            img = None
            st.info("Upload or select an image to continue.")

    with col2:
        st.subheader("Predictions")

        if img is None:
            st.write("Waiting for image...")

        elif pred_btn:

            # YOLO
            if yolo_model:
                try:
                    res = yolo_model(img)
                    probs = res[0].probs.data.tolist()
                    names = res[0].names
                    i = int(np.argmax(probs))
                    conf = float(probs[i])

                    st.markdown("### YOLOv8")
                    st.write(f"Prediction: **{names[i].upper()}**")
                    st.write(f"Confidence: **{conf:.2f}**")
                    st.progress(conf)

                except Exception as e:
                    st.error(f"YOLO Error: {e}")

            # Generic classifier
            def run_classifier(model, tf, title):
                try:
                    t = tf(img).unsqueeze(0)
                    with torch.no_grad():
                        out = model(t)
                    p = torch.softmax(out,1).numpy()[0]
                    idx = int(np.argmax(p))
                    label = ["no_tumor","tumor"][idx]
                    conf = float(p[idx])

                    st.markdown(f"### {title}")
                    st.write(f"Prediction: **{label.upper()}**")
                    st.write(f"Confidence: **{conf:.2f}**")
                    st.progress(conf)

                except Exception as e:
                    st.error(str(e))

            run_classifier(faster_model, faster_tf, "Faster R-CNN")
            run_classifier(mask_model,   mask_tf,   "Mask R-CNN")
            run_classifier(retina_model, retina_tf, "RetinaNet")
            run_classifier(ssd_model,    ssd_tf,    "SSD-MobileNet")


# ------------------------------------------------------------
# EVALUATION PAGE  — FULL VERSION (with tables, charts, best model)
# ------------------------------------------------------------
else:

    st.title("Model Evaluation & Comparison")

    if not METRICS_CSV.exists():
        st.error("metrics_summary.csv missing.")
    else:
        df = pd.read_csv(METRICS_CSV)

        # ---------------- SUMMARY TABLE ----------------
        st.subheader("Summary Table")
        st.dataframe(df)

        st.download_button(
            "Download metrics_summary.csv",
            df.to_csv(index=False).encode(),
            "metrics_summary.csv"
        )

        # ---------------- BAR CHART ----------------
        st.subheader("Accuracy / Precision / Recall / F1 Comparison")

        metrics = ["accuracy","precision","recall","f1_score"]

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, m in enumerate(metrics):
            r, c = divmod(i, 2)
            ax[r][c].bar(df["model"], df[m], color="#2e86de")
            ax[r][c].set_title(m.upper())
            for j, val in enumerate(df[m]):
                ax[r][c].text(j, val + 0.1, f"{val:.2f}", ha='center')

        plt.tight_layout()
        st.pyplot(fig)

        # ---------------- CONFUSION MATRICES ----------------
        st.write("---")
        st.subheader("Confusion Matrices")

        cm_files = sorted(RESULTS_DIR.glob("cm_*.png"))
        cols = st.columns(2)

        for i, cm in enumerate(cm_files):
            with cols[i % 2]:
                st.image(str(cm), caption=cm.name)

        # ---------------- PER-MODEL CSV RESULTS ----------------
        st.write("---")
        st.subheader("Per-model CSV results (example rows)")

        for model in df["model"].tolist():
            f = RESULTS_DIR / f"{model}_results.csv"
            if f.exists():
                st.write(f"### {model}")
                small = pd.read_csv(f).head(6)
                st.dataframe(small)
                st.write("")

        # ---------------- BEST MODEL ----------------
        st.write("---")
        best_row = df.loc[df["accuracy"].idxmax()]
        st.success(
            f"Best model by accuracy: **{best_row['model']}** — accuracy {best_row['accuracy']}"
        )

    st.write("<hr>", unsafe_allow_html=True)
    st.markdown(
        "Built for project: **Deep Learning for Brain Tumor Detection** — Models: YOLOv8, Faster R-CNN, Mask R-CNN, RetinaNet, SSD-MobileNet"
    )
