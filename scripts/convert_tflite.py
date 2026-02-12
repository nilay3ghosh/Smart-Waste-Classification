# scripts/convert_tflite.py
import tensorflow as tf
MODEL_PATH = "models/resnet_garbage.keras"
TFLITE_PATH = "models/resnet_garbage.tflite"

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Optional: float16 quantization
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print("Wrote", TFLITE_PATH)
