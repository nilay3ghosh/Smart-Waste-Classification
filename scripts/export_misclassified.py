# scripts/export_misclassified.py
import json, os, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy2

MODEL_PATH = "models/resnet_garbage.keras"
TEST_DIR = "data/test"
OUT = "debug_misclassified"
IMG_SIZE=(224,224)
BATCH_SIZE=16

os.makedirs(OUT, exist_ok=True)
model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

preds = model.predict(test_gen, verbose=1)
y_true = test_gen.classes
y_pred = np.argmax(preds, axis=1)

with open("mapping/classes.json") as f:
    class_indices = json.load(f)
inv = {v:k for k,v in class_indices.items()}
class_names = [inv[i] for i in range(len(inv))]

# copy misclassified images to debug folder
filenames = np.array(test_gen.filenames)
for i,(true_idx,pred_idx) in enumerate(zip(y_true,y_pred)):
    if true_idx != pred_idx:
        true_cls = class_names[true_idx]
        pred_cls = class_names[pred_idx]
        out_dir = os.path.join(OUT, f"{true_cls}_as_{pred_cls}")
        os.makedirs(out_dir, exist_ok=True)
        src = os.path.join(TEST_DIR, filenames[i])
        copy2(src, out_dir)

print("Copied misclassified images to", OUT)
