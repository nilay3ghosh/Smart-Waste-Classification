# scripts/train_resnet.py
import os, json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = (224,224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 6
MODEL_OUT = "models/resnet_garbage.keras"
FINE_TUNE_AT = 140

os.makedirs("models", exist_ok=True)
os.makedirs("mapping", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

num_classes = len(train_gen.class_indices)
print("Found classes:", train_gen.class_indices)

# Save class mapping files for inference
with open("mapping/classes.json", "w") as f:
    json.dump(train_gen.class_indices, f)
inv = {v:k for k,v in train_gen.class_indices.items()}
with open("mapping/classes.txt", "w") as f:
    for i in range(len(inv)):
        f.write(inv[i] + "\n")

# Build model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE+(3,))
base_model.trainable = False

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

cb = [
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(train_gen, epochs=INITIAL_EPOCHS, validation_data=val_gen, callbacks=cb)

# Fine-tune
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_f = model.fit(train_gen, epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
                      initial_epoch=history.epoch[-1], validation_data=val_gen, callbacks=cb)

model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)
