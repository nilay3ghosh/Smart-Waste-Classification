# scripts/infer_webcam_object_only.py
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import time
import os

# -------- CONFIG --------
MODEL_PATH = "models/resnet_garbage.keras"
CLASSES_FILE = "mapping/classes.txt"
MAPPING_CSV = "mapping/class_to_biodeg.csv"
IMG_SIZE = (224, 224)
MIN_AREA = 2000            # minimum contour area (tweak)
BG_CAPTURE_FRAMES = 30     # number of frames to average when capturing background
BLUR_KERNEL = (5,5)
THRESH = 30                # threshold for diff image
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
SHOW_RAW_MASK = False      # set True to debug mask
# ------------------------

# Load model & classes
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_FILE) as f:
    class_names = [line.strip() for line in f if line.strip()]

df = pd.read_csv(MAPPING_CSV, header=None, names=['class','biodeg'])
biodeg_map = dict(zip(df['class'].str.strip(), df['biodeg'].str.strip()))

def map_to_biodeg(cls):
    return biodeg_map.get(cls, "unknown")

# Helper: preprocess crop for model
def preprocess_crop(crop):
    img = cv2.resize(crop, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam (0)")

bg_frame = None
bg_captured = False

print("Instructions:")
print("  - Press 'b' to capture background (make sure no object in front).")
print("  - Press 'r' to reset background capture.")
print("  - Press 'q' to quit.")
print("  - Press 'c' to print top-3 preds to console.")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_disp = frame.copy()
    h, w = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF

    # Capture background (average over several frames for stability)
    if key == ord('b'):
        print("Capturing background... hold still")
        acc = None
        count = 0
        for i in range(BG_CAPTURE_FRAMES):
            ret2, f2 = cap.read()
            if not ret2:
                break
            gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype('float32')
            if acc is None:
                acc = gray
            else:
                cv2.accumulate(gray, acc)
            count += 1
            time.sleep(0.02)
        if count > 0:
            avg = (acc / float(count)).astype('uint8')
            bg_frame = avg
            bg_captured = True
            print("Background captured.")
        else:
            print("Failed to capture background.")

    if key == ord('r'):
        bg_frame = None
        bg_captured = False
        print("Background reset.")

    # If background captured, compute foreground mask
    object_found = False
    label = "none"
    prob = 0.0
    biodeg_tag = "unknown"

    if bg_captured and bg_frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # smooth both images
        gray_blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        bg_blur = cv2.GaussianBlur(bg_frame, BLUR_KERNEL, 0)

        # absolute difference
        diff = cv2.absdiff(bg_blur, gray_blur)
        _, mask = cv2.threshold(diff, THRESH, 255, cv2.THRESH_BINARY)

        # morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
        mask = cv2.dilate(mask, MORPH_KERNEL, iterations=2)

        # optional: show mask for debugging
        if SHOW_RAW_MASK:
            cv2.imshow("mask_debug", mask)

        # find contours; pick largest by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # sort by area desc
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA:
                    continue
                x,y,wc,hc = cv2.boundingRect(cnt)
                # enlarge bbox slightly for safe margin
                pad = 10
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(frame.shape[1], x+wc+pad); y1 = min(frame.shape[0], y+hc+pad)
                crop = frame[y0:y1, x0:x1]
                # classify
                x_in = preprocess_crop(crop)
                preds = model.predict(x_in)
                idx = preds[0].argmax()
                prob = float(preds[0][idx])
                label = class_names[idx]
                biodeg_tag = map_to_biodeg(label)
                object_found = True

                # draw bounding box and label
                color = (0,255,0) if "bio" in biodeg_tag.lower() else (0,0,255) if biodeg_tag.lower().startswith("non") else (0,165,255)
                cv2.rectangle(frame_disp, (x0,y0), (x1,y1), color, 2)
                cv2.putText(frame_disp, f"{label} ({prob*100:.1f}%)", (x0, y0-10), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_disp, biodeg_tag.upper(), (x0, y1+25), font, 0.7, color, 2, cv2.LINE_AA)
                # only use the largest valid contour
                break

    # draw instructions & background state
    status = "BG captured" if bg_captured else "No BG (press 'b')"
    cv2.putText(frame_disp, status, (10,30), font, 0.7, (200,200,200), 2, cv2.LINE_AA)

    cv2.imshow("Object-only Classifier", frame_disp)

    if key == ord('q'):
        break
    if key == ord('c'):
        # print debug
        print(f"Label: {label}, Prob: {prob:.4f}, Biodeg: {biodeg_tag}")

cap.release()
cv2.destroyAllWindows()
