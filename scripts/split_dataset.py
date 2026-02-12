# scripts/split_dataset.py
import os, shutil, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data_raw/dataset-resized", help="source folder containing class folders")
parser.add_argument("--out", default="data", help="output base folder")
parser.add_argument("--train_frac", type=float, default=0.7)
parser.add_argument("--val_frac", type=float, default=0.15)
parser.add_argument("--test_frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6

random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
for split in ("train","val","test"):
    os.makedirs(os.path.join(args.out, split), exist_ok=True)

for classname in os.listdir(args.source):
    src_class_dir = os.path.join(args.source, classname)
    if not os.path.isdir(src_class_dir):
        continue
    imgs = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]
    for split_name, split_list in (("train",train_imgs),("val",val_imgs),("test",test_imgs)):
        out_dir = os.path.join(args.out, split_name, classname)
        os.makedirs(out_dir, exist_ok=True)
        for fname in split_list:
            src = os.path.join(src_class_dir, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(src, dst)
    print(f"Class {classname}: total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
print("Done splitting.")
