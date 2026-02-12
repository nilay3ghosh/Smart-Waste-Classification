# ♻️ Smart Waste Classification using Deep Learning

> An AI-powered Computer Vision system for automated waste segregation using ResNet (CNN) and YOLOv5.

---

## 🚀 Project Summary

Developed a Deep Learning-based waste classification system capable of automatically identifying and categorizing waste materials using image data.  

The system leverages **Transfer Learning (ResNet architecture)** and **YOLOv5 object detection** to improve classification accuracy and enable real-time prediction.

This project demonstrates strong understanding of:
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Model Evaluation Metrics
- Real-time Inference
- Computer Vision Pipelines
- ML Project Structuring & Deployment Readiness

---

## 🎯 Problem Statement

Manual waste segregation is inefficient and error-prone.  
This project automates waste classification to support:

- Smart Recycling Systems
- Sustainable Waste Management
- Smart City Infrastructure
- Environmental Monitoring Solutions

---

## 🧠 Technical Implementation

### 🔹 1. Data Pipeline
- Dataset validation & cleaning
- Automated train-validation split
- Class distribution verification
- Data preprocessing & normalization

### 🔹 2. Model Architecture
- Transfer Learning using **ResNet**
- Fine-tuning last layers
- Dropout regularization
- Optimized learning rate scheduling

### 🔹 3. Training Strategy
- Cross-entropy loss
- Adam optimizer
- Early stopping
- Model checkpoint saving

### 🔹 4. Evaluation Metrics
- Accuracy
- Precision & Recall
- F1-Score
- Confusion Matrix
- ROC Curve

### 🔹 5. Real-Time Inference
- OpenCV-based webcam prediction
- Live classification display
- YOLOv5 integration for object detection-based waste identification

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras, PyTorch |
| Computer Vision | OpenCV |
| ML Utilities | NumPy, Scikit-learn, Matplotlib |
| Model | ResNet (Transfer Learning), YOLOv5 |

---

## 📂 Project Structure

Smart-Waste-Classification/
│
├── scripts/
│ ├── train_resnet_improved.py
│ ├── evaluate.py
│ ├── infer_webcam_model.py
│ ├── split_dataset.py
│ └── check_dataset.py
│
├── data/ (ignored in GitHub)
├── models/ (ignored in GitHub)
├── yolov5/ (external dependency)
└── README.md


---

## ⚙️ How to Run

### Clone Repository
git clone https://github.com/nilay3ghosh/Smart-Waste-Classification.git
cd Smart-Waste-Classification


### Setup Environment
conda create -n smartwaste python=3.10
conda activate smartwaste
pip install -r requirements.txt


### Train Model
python scripts/train_resnet_improved.py


### Evaluate Model
python scripts/evaluate.py


### Run Real-Time Inference
python scripts/infer_webcam_model.py


---

## 📈 Key Highlights (Resume Points)

- Built end-to-end Deep Learning pipeline for image classification
- Implemented Transfer Learning using ResNet for improved accuracy
- Integrated YOLOv5 for object detection-based classification
- Designed modular and scalable ML project structure
- Enabled real-time webcam inference using OpenCV
- Applied model evaluation techniques for performance validation

---

## 🌍 Real-World Applications

- Smart Dustbins
- AI-powered Recycling Plants
- Sustainable Waste Management Systems
- Smart City Projects

---

## 👨‍💻 Author

**Nilay Ghosh**  
Machine Learning & Computer Vision Enthusiast  
GitHub: https://github.com/nilay3ghosh  

---

⭐ If you found this project useful, consider giving it a star!
Run:

git add README.md
git commit -m "Updated README - resume focused version"
git push
