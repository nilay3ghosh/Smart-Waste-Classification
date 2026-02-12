# scripts/train_resnet_improved.py
import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt   # optional plotting

# --------- Config ----------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224, 224)
BATCH_SIZE = 24

# Training schedule (recommended)
HEAD_EPOCHS = 30
FINE_TUNE_EPOCHS = 40
FINE_TUNE_AT = 50   # unfreeze from this layer onward when fine-tuning

MODEL_OUT = "models/resnet_garbage.keras"

os.makedirs("models", exist_ok=True)
os.makedirs("mapping", exist_ok=True)

# --------- Data generators (richer augmentation on train) ----------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.12,
    zoom_range=0.2,
    brightness_range=(0.6, 1.4),
    channel_shift_range=20.0,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --------- class info & mapping ----------
num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

with open("mapping/classes.json", "w") as f:
    json.dump(train_gen.class_indices, f)

inv = {v: k for k, v in train_gen.class_indices.items()}
with open("mapping/classes.txt", "w") as f:
    for i in range(len(inv)):
        f.write(inv[i] + "\n")

# --------- class weights to handle imbalance ----------
y = train_gen.classes
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = {i: w for i, w in enumerate(cw)}
print("Class weights:", class_weights)

# --------- steps per epoch (explicit) ----------
steps_per_epoch = int(math.ceil(len(train_gen.filenames) / float(BATCH_SIZE)))
validation_steps = int(math.ceil(len(val_gen.filenames) / float(BATCH_SIZE)))
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

# --------- Build model (ResNet50 backbone) ----------
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------- Callbacks ----------
cb = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# --------- Step 1: Train head (base frozen) ----------
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=HEAD_EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=cb,
    class_weight=class_weights
)

# --------- Unfreeze for fine-tuning ----------
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------- Step 2: Fine-tune (resume without repeating last epoch) ----------
initial_epoch = len(history.epoch)   # <-- avoids duplicate epoch
total_epochs = HEAD_EPOCHS + FINE_TUNE_EPOCHS

history_f = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=total_epochs,
    initial_epoch=initial_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=cb,
    class_weight=class_weights
)

# --------- Save final model ----------
model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)

