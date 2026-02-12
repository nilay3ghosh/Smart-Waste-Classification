# scripts/check_dataset.py
import os
from collections import Counter
src = "data_raw/dataset-resized"  # or data/train if already split
for cls in sorted(os.listdir(src)):
    p = os.path.join(src, cls)
    if os.path.isdir(p):
        cnt = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        print(f"{cls:12s} -> {cnt}")
