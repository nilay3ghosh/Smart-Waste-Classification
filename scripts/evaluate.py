# evaluate.py
import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -------- CONFIG --------
MODEL_PATH = "models/resnet_garbage.keras"    # trained model
TEST_DIR = "data/test"                        # test images
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs("models", exist_ok=True)

# -------- LOAD MODEL --------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# -------- TEST DATA GENERATOR --------
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -------- PREDICTIONS --------
y_true = test_gen.classes
y_prob = model.predict(test_gen)
y_pred = np.argmax(y_prob, axis=1)

# -------- LOAD CLASS NAMES --------
with open("mapping/classes.json") as f:
    class_indices = json.load(f)  # {"cardboard": 0, "glass": 1, ...}

# Invert: 0 -> "cardboard", 1 -> "glass", etc.
inv_map = {v: k for k, v in class_indices.items()}

# Create class_names list in order
class_names = [inv_map[i] for i in range(len(inv_map))]
print("Class names:", class_names)

# -------- CLASSIFICATION REPORT --------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

# -------- ROC–AUC (MULTICLASS) --------
n_classes = len(class_names)
y_true_bin = label_binarize(y_true, classes=range(n_classes))

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curves (Multiclass)")
plt.legend()
plt.tight_layout()
plt.savefig("models/roc_auc.png")
plt.show()


