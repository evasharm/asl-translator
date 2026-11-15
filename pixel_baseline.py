"""
Pixel-based baseline, used when landmark extraction is not possible (e.g. the
Kaggle 'Hand Signs' set, whose images already carry a drawn skeleton).

  baseline  48x48 grayscale pixels + logistic regression
  iter1     HOG features (skimage) + linear SVM

Same contiguous-block split as train_baseline.py (last 20% per class in
filename order).
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load(data_dir, size):
    X, y, idx = [], [], []
    for cls in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            continue
        files = sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMAGE_EXTS))
        for i, p in enumerate(files):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
            y.append(cls)
            idx.append(i)
    return np.stack(X), np.array(y), np.array(idx)


def hog_feats(imgs):
    from skimage.feature import hog
    return np.stack([hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for im in imgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--out", default="results_pixel")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, idx = load(args.data, args.size)
    classes = sorted(set(y))
    test = np.zeros(len(y), bool)
    for c in classes:
        m = np.where(y == c)[0]
        m = m[np.argsort(idx[m])]
        test[m[-int(round(len(m) * args.test_frac)):]] = True
    tr = ~test
    print(f"{len(y)} images, {len(classes)} classes, train {tr.sum()} test {test.sum()}")

    results = {}
    exps = {
        "baseline": ("pixels", X.reshape(len(X), -1).astype(np.float32) / 255.0,
                     make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))),
        "iter1": ("hog", hog_feats(X), make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=5000))),
    }
    for name, (feat, F, model) in exps.items():
        t0 = time.time()
        model.fit(F[tr], y[tr])
        pred = model.predict(F[test])
        acc = accuracy_score(y[test], pred)
        cm = confusion_matrix(y[test], pred, labels=classes)
        off = cm.copy(); np.fill_diagonal(off, 0)
        pairs = []
        for _ in range(5):
            i, j = np.unravel_index(off.argmax(), off.shape)
            if off[i, j] == 0:
                break
            pairs.append((classes[i], classes[j], int(off[i, j]))); off[i, j] = 0
        results[name] = dict(features=feat, dim=int(F.shape[1]), accuracy=float(acc),
                             fit_seconds=time.time() - t0, confused_pairs=pairs,
                             n_train=int(tr.sum()), n_test=int(test.sum()))
        print(f"[{name}] {feat} dim={F.shape[1]} acc={acc * 100:.1f}% fit={time.time() - t0:.1f}s pairs={pairs}")
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
