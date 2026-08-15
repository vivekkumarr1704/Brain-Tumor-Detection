# 🧠 Brain Tumor Detection Using Deep Learning

A comparative deep learning framework for brain tumor prediction from MRI images using five different deep learning architectures:

- YOLOv8
- Faster R-CNN
- Mask R-CNN
- RetinaNet
- SSD-MobileNet

The project provides an interactive Streamlit interface for MRI-based prediction and model evaluation.

---

## 🚀 Live Demo

🔗 **Live Application:**  
`PASTE_YOUR_STREAMLIT_LIVE_LINK_HERE`

> The live application link will be added after deployment.

---

## 📌 Project Overview

Brain tumor detection from MRI images is an important medical imaging task where accurate identification of tumor and non-tumor cases can support computer-aided analysis.

This project develops a controlled comparative framework in which five deep learning architectures are evaluated using the same MRI dataset, preprocessing pipeline, and evaluation metrics.

The main objective is to determine which architecture provides the strongest performance for **Tumor vs No Tumor** MRI classification.

---

## 🎯 Objectives

- Develop a deep learning-based brain tumor prediction system.
- Compare five different deep learning architectures under consistent conditions.
- Evaluate model performance using standard classification metrics.
- Identify the best-performing architecture.
- Provide an interactive interface for MRI prediction and model evaluation.
- Maintain a structured and reproducible research workflow.

---

## 🗂️ Dataset

The project uses publicly accessible brain MRI image collections, including Kaggle-based datasets.

### Classes

- **Tumor**
- **No Tumor**

### Dataset Split

| Dataset | Percentage |
|---------|------------|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

---

## 🔄 Image Preprocessing

The MRI images are processed before being provided to the deep learning models.

### Preprocessing Pipeline

```text
Raw MRI Image
      ↓
Quality Check
      ↓
Image Resizing
      ↓
Normalization
      ↓
RGB Conversion
      ↓
Model Input
Input Dimensions
CNN-based models: 224 × 224
YOLOv8: 640 × 640

No manual feature engineering is required because the deep learning architectures automatically learn hierarchical image features from the input images.

🧠 Models Evaluated
1. YOLOv8
One-stage architecture
Fast inference
Efficient prediction
Suitable for real-time applications
2. Faster R-CNN
Region Proposal Network
ResNet50-based feature extraction
Strong spatial feature learning
3. Mask R-CNN
Region-based architecture
Additional mask branch
Detailed spatial feature extraction
Strong discriminative capability
4. RetinaNet
One-stage architecture
Feature Pyramid Network
Focal-loss-based design
Effective multi-scale representation
5. SSD-MobileNet
Lightweight architecture
MobileNetV2 backbone
Computationally efficient
Suitable for low-resource environments
⚙️ Proposed Methodology

The complete framework follows a common pipeline for all five architectures:

             MRI Dataset
          (Tumor / No Tumor)
                   ↓
            Preprocessing
          (Resize + Normalize)
                   ↓
       ┌───────────┼───────────┐
       ↓           ↓           ↓
    YOLOv8     Faster R-CNN  Mask R-CNN
       ↓           ↓           ↓
   RetinaNet   SSD-MobileNet
       └───────────┼───────────┘
                   ↓
       Evaluation & Comparison
                   ↓
 Accuracy | Precision | Recall | F1
                   ↓
          Best Performing Model

All architectures are evaluated using the same overall experimental framework to enable a fair comparison.

🛠️ Technology Stack
Python
PyTorch
Torchvision
Ultralytics YOLOv8
OpenCV
NumPy
Pandas
Matplotlib
Streamlit
Scikit-learn
📊 Evaluation Metrics

The models are evaluated using:

Accuracy

Measures the overall percentage of correctly classified images.

Precision

Measures the reliability of predicted tumor cases.

Recall / Sensitivity

Measures the ability to correctly identify actual tumor-positive cases.

F1-Score

Provides a balance between precision and recall.

Confusion Matrix

Provides a detailed view of correct and incorrect classifications, including false positives and false negatives.

📈 Experimental Results

The comparative evaluation produced the following results:

Model	Accuracy	Precision	Recall	F1-Score
Mask R-CNN	99.33%	99.33%	99.34%	99.33%
YOLOv8	99.11%	99.11%	99.11%	99.11%
Faster R-CNN	99.11%	99.12%	99.11%	99.11%
RetinaNet	98.88%	98.89%	98.89%	98.88%
SSD-MobileNet	98.66%	98.70%	98.66%	98.66%
🏆 Best Performing Model

Mask R-CNN achieved the highest overall performance in the comparative study:

Accuracy: 99.33%
Precision: 99.33%
Recall: 99.34%
F1-Score: 99.33%

YOLOv8 and Faster R-CNN were highly competitive, while RetinaNet and SSD-MobileNet also achieved accuracy above 98%.

💡 Why Mask R-CNN Performed Best

Mask R-CNN demonstrated the strongest overall performance among the evaluated architectures.

Its region-based architecture and additional mask branch provide strong spatial feature representation, which can be useful for capturing detailed structural information in MRI images.

However, the reported performance represents results obtained under the defined experimental setup and dataset. Further evaluation on larger and independent datasets would be required to assess generalization.

🖥️ Streamlit Application

The project includes an interactive Streamlit interface that allows users to:

Upload an MRI image.
Run predictions using the available deep learning models.
View model predictions and confidence scores.
Compare model outputs.
View evaluation results and performance metrics.
Run the Application
streamlit run interface/app.py
📁 Project Structure
Brain-Tumor-Detection/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── interface/
│   ├── app.py
│   └── templates/
│
├── models/
│   ├── faster_rcnn/
│   ├── mask_rcnn/
│   ├── retinanet/
│   ├── ssd/
│   └── YOLOv8/
│
├── results/
│   ├── confusion_matrix/
│   ├── metrics/
│   └── predictions/
│
├── data.yaml
├── evaluate_models.py
├── requirements.txt
├── README.md
└── .gitignore

The local virtual environment and dataset are excluded from the Git repository through .gitignore.

🔧 Installation

Clone the repository:

git clone https://github.com/vivekkumarr1704/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

Create and activate a virtual environment:

python3 -m venv brainenv
source brainenv/bin/activate

Install the required dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run interface/app.py
🔬 Research Contribution

The primary contribution of this project is a controlled comparative evaluation of five deep learning architectures for brain tumor prediction using MRI images.

Instead of comparing results from unrelated datasets or experimental conditions, the framework evaluates the selected architectures within a common workflow, making the comparison more consistent.

🔮 Future Scope

Future development can focus on:

Testing on larger and more diverse MRI datasets.
Extending binary classification to multi-class tumor classification.
Tumor localization and segmentation.
Explainable AI for better interpretation of model predictions.
Evaluation on independent clinical datasets.
Improving model efficiency for real-time applications.
⚠️ Disclaimer

This project is developed for academic and research purposes.

The system is intended as a computer-aided research prototype and should not be considered a replacement for professional medical diagnosis or clinical decision-making.

The reported results are specific to the dataset and experimental setup used in this study.

👨‍💻 Authors

Vivek Kumar

Galgotias University
📄 Research Work

Applications of Deep Learning in Brain Disease Prediction

This project was developed as part of the Final-Year Capstone Project and research work at Galgotias University.



📜 License

This project is intended for academic and research purposes.




