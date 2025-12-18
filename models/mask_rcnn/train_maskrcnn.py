# interface/app.py
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Brain Tumor Detection (5-models)", layout="wide")

YOLO_PATH = "runs/classify/train2/weights/best.pt"              # your yolov8 classify best
FASTER_RCNN_PATH = "models/faster_rcnn/faster_rcnn_classifier.pth"
RETINANET_PATH = "models/retinanet/retinanet_classifier.pth"
SSD_PATH = "models/ssd/ssd_mobilenet_classifier.pth"
MASKRCNN_PATH = "models/mask_rcnn/maskrcnn_classifier.pth"
TEST_FOLDER = "dataset/test"


# --------------------------
# Load YOLO (classification)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO(YOLO_PATH)

yolo_model = load_yolo()


# --------------------------
# Generic loader: ResNet50 classifier
# --------------------------
def _load_resnet_classifier(path):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return model, transform

# wrappers with caching
@st.cache_resource(show_spinner=False)
def load_faster():
    return _load_resnet_classifier(FASTER_RCNN_PATH)

@st.cache_resource(show_spinner=False)
def load_retina():
    return _load_resnet_classifier(RETINANET_PATH)

@st.cache_resource(show_spinner=False)
def load_maskrcnn():
    return _load_resnet_classifier(MASKRCNN_PATH)

# SSD uses MobileNetV2 backbone
@st.cache_resource(show_spinner=False)
def load_ssd():
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(SSD_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return model, transform

# load all
faster_model, faster_transform = load_faster()
retina_model, retina_transform = load_retina()
mask_model, mask_transform = load_maskrcnn()
ssd_model, ssd_transform = load_ssd()


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("Controls")
sample_list = []
if os.path.isdir(TEST_FOLDER):
    for cls in sorted(os.listdir(TEST_FOLDER)):
        folder = Path(TEST_FOLDER) / cls
        if folder.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                sample_list.extend([str(p) for p in folder.glob(ext)])

selected_sample = st.sidebar.selectbox("Choose sample image", ["-- none --"] + sample_list)

if st.sidebar.button("Reload Models"):
    load_yolo.clear()
    load_faster.clear()
    load_retina.clear()
    load_maskrcnn.clear()
    load_ssd.clear()
    st.sidebar.success("Models reloaded")


# --------------------------
# HEADER
# --------------------------
st.markdown("<h1 style='font-size:42px;'>Brain Tumor Detection — 5 Models</h1>", unsafe_allow_html=True)
st.write("YOLOv8 (classification) + Faster R-CNN + Mask R-CNN + RetinaNet + SSD-MobileNet (all classifier versions).")
st.write("---")


# --------------------------
# Upload area
# --------------------------
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if selected_sample != "-- none --":
        uploaded = selected_sample

    if uploaded:
        if isinstance(uploaded, str):
            img = Image.open(uploaded)
            size_kb = Path(uploaded).stat().st_size // 1024
        else:
            img = Image.open(uploaded)
            size_kb = uploaded.size // 1024

        img_display = ImageOps.fit(img.convert("RGB"), (500, 500))
        st.image(img_display, caption=f"Uploaded Image ({size_kb} KB)")
        predict_btn = st.button("Predict")
    else:
        img = None
        st.info("Upload or select sample image.")


# --------------------------
# Predictions (right column)
# --------------------------
with right:
    st.subheader("Predictions")

    if img and predict_btn:
        # YOLOv8
        res = yolo_model(img)
        probs = res[0].probs.data.tolist()
        names = res[0].names
        idx = int(np.argmax(probs))
        y_class = names[idx]
        y_conf = float(probs[idx])

        st.markdown("### YOLOv8 Result")
        st.write(f"Prediction: **{y_class.upper()}**")
        st.write(f"Confidence: **{y_conf:.2f}**")
        st.progress(y_conf)
        st.write("---")

        # Faster R-CNN (ResNet50 classifier)
        t_f = faster_transform(img).unsqueeze(0)
        with torch.no_grad():
            out_f = faster_model(t_f)
            pf = torch.softmax(out_f, dim=1).numpy()[0]
        f_idx = int(np.argmax(pf))
        f_class = ["no_tumor", "tumor"][f_idx]
        f_conf = float(pf[f_idx])

        st.markdown("### Faster R-CNN Result")
        st.write(f"Prediction: **{f_class.upper()}**")
        st.write(f"Confidence: **{f_conf:.2f}**")
        st.progress(f_conf)
        st.write("---")

        # Mask R-CNN (classifier-style)
        t_m = mask_transform(img).unsqueeze(0)
        with torch.no_grad():
            out_m = mask_model(t_m)
            pm = torch.softmax(out_m, dim=1).numpy()[0]
        m_idx = int(np.argmax(pm))
        m_class = ["no_tumor", "tumor"][m_idx]
        m_conf = float(pm[m_idx])

        st.markdown("### Mask R-CNN Result")
        st.write(f"Prediction: **{m_class.upper()}**")
        st.write(f"Confidence: **{m_conf:.2f}**")
        st.progress(m_conf)
        st.write("---")

        # RetinaNet
        t_r = retina_transform(img).unsqueeze(0)
        with torch.no_grad():
            out_r = retina_model(t_r)
            pr = torch.softmax(out_r, dim=1).numpy()[0]
        r_idx = int(np.argmax(pr))
        r_class = ["no_tumor", "tumor"][r_idx]
        r_conf = float(pr[r_idx])

        st.markdown("### RetinaNet Result")
        st.write(f"Prediction: **{r_class.upper()}**")
        st.write(f"Confidence: **{r_conf:.2f}**")
        st.progress(r_conf)
        st.write("---")

        # SSD-MobileNet
        t_s = ssd_transform(img).unsqueeze(0)
        with torch.no_grad():
            out_s = ssd_model(t_s)
            ps = torch.softmax(out_s, dim=1).numpy()[0]
        s_idx = int(np.argmax(ps))
        s_class = ["no_tumor", "tumor"][s_idx]
        s_conf = float(ps[s_idx])

        st.markdown("### SSD-MobileNet Result")
        st.write(f"Prediction: **{s_class.upper()}**")
        st.write(f"Confidence: **{s_conf:.2f}**")
        st.progress(s_conf)

    else:
        st.write("Waiting for image...")


st.write("---")
st.subheader("Next: final evaluation charts (accuracy, precision, recall, F1) and export.")
