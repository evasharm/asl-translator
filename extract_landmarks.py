"""
Run MediaPipe HandLandmarker over the Kaggle ASL Alphabet dataset and cache the
landmarks, so training never has to touch the images again.

Dataset layout (after `kaggle datasets download -d grassknoted/asl-alphabet --unzip`):
  data/asl_alphabet_train/asl_alphabet_train/<LETTER>/<LETTER><idx>.jpg   idx = 1..3000
  data/asl_alphabet_test/asl_alphabet_test/<LETTER>_test.jpg

The 3000 images per class are consecutive video frames of one signer, so a random
split would leak near-duplicate frames into the test set. We instead take every
k-th frame (--per-class controls how many) and later split by frame index.

Output: a single .npz with
  landmarks   (N, 21, 3)  float32, zeros where no hand was detected
  detected    (N,)        bool
  handedness  (N,)        'Left' / 'Right' / ''
  score       (N,)        hand presence score
  label       (N,)        letter string
  index       (N,)        frame index inside the class
  path        (N,)        image path

Usage:
  python extract_landmarks.py --data data/asl_alphabet_train/asl_alphabet_train \
      --per-class 300 --out cache/train_landmarks.npz
  python extract_landmarks.py --data data/asl_alphabet_test/asl_alphabet_test \
      --flat --out cache/test_landmarks.npz
"""

import argparse
import glob
import os
import re
import sys
import time
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.environ.get("HAND_MODEL", "models/hand_landmarker.task")
_landmarker = None  # one per worker process
_min_conf = 0.5
_pad = 0.0


def _init_worker(min_conf, pad):
    global _landmarker, _min_conf, _pad
    _min_conf = min_conf
    _pad = pad
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    _landmarker = vision.HandLandmarker.create_from_options(options)


def _process(item):
    path, label, index = item
    img = cv2.imread(path)
    if img is None:
        return path, label, index, np.zeros((21, 3), np.float32), False, "", 0.0
    h0, w0 = img.shape[:2]
    if _pad > 0:
        # Add a border so the hand does not fill the frame; MediaPipe's palm
        # detector expects some context around the hand. Landmarks are mapped
        # back to the original image's normalized coordinates below.
        p = int(_pad * max(img.shape[:2]))
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return path, label, index, np.zeros((21, 3), np.float32), False, "", 0.0
    lm = np.array([(p.x, p.y, p.z) for p in res.hand_landmarks[0]], np.float32)
    if _pad > 0:
        # undo the border: padded normalized coords -> original normalized coords
        H, W = img.shape[:2]
        p = (H - h0) / 2  # same p on all sides
        lm[:, 0] = (lm[:, 0] * W - p) / w0
        lm[:, 1] = (lm[:, 1] * H - p) / h0
    hand = res.handedness[0][0]
    return path, label, index, lm, True, hand.category_name, float(hand.score)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _images_in(d):
    return [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if p.lower().endswith(IMAGE_EXTS)]


def list_images(data_dir, per_class, flat):
    """Return (path, label, index) triples. index is the frame number parsed from
    the filename when present, else the position in sorted order, so that a
    contiguous-block split still separates neighbouring frames."""
    items = []
    if flat:
        for p in sorted(_images_in(data_dir)):
            label = re.split(r"[_\-]", os.path.basename(p))[0]
            items.append((p, label, 0))
        return items
    for cls in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, cls)
        if not os.path.isdir(cdir):
            continue
        files = []
        for p in _images_in(cdir):
            # A123.jpg -> 123; timestamp names like image_1681446027.28.jpg -> sorted position
            m = re.match(r"^[A-Za-z_\-]*(\d+)\.[A-Za-z]+$", os.path.basename(p))
            files.append((int(m.group(1)) if m else -1, p))
        files.sort(key=lambda t: (t[0], t[1]))
        files = [(i if n < 0 else n, p) for i, (n, p) in enumerate(files)]
        if per_class and per_class < len(files):
            step = len(files) / per_class
            files = [files[int(i * step)] for i in range(per_class)]
        items += [(p, cls, idx) for idx, p in files]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--flat", action="store_true", help="dataset is a flat folder of <LABEL>_*.jpg")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.0,
                    help="black border added around each image, as a fraction of its longer side")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Missing {MODEL_PATH}. Download hand_landmarker.task first (see README).")

    items = list_images(args.data, args.per_class, args.flat)
    print(f"{len(items)} images, {len(set(i[1] for i in items))} classes, {args.workers} workers")
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.min_conf, args.pad)) as pool:
        rows = []
        for k, r in enumerate(pool.imap(_process, items, chunksize=16), 1):
            rows.append(r)
            if k % 500 == 0 or k == len(items):
                print(f"  {k}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    paths, labels, idxs, lms, det, hands, scores = zip(*rows)
    signer = [(re.match(r"^(hand\d+)", os.path.basename(p)) or [""])[0] for p in paths]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        landmarks=np.stack(lms), detected=np.array(det), handedness=np.array(hands),
        score=np.array(scores, np.float32), label=np.array(labels),
        index=np.array(idxs), path=np.array(paths), signer=np.array(signer),
    )
    det = np.array(det)
    print(f"\nSaved {args.out}")
    print(f"Detection rate: {det.mean() * 100:.1f}%  ({det.sum()}/{len(det)})")
    print(f"Throughput:     {len(items) / elapsed:.1f} images/s on {args.workers} workers")
    labels = np.array(labels)
    per_class = {}
    print("Per-class detection rate:")
    for cls in sorted(set(labels)):
        m = labels == cls
        per_class[cls] = float(det[m].mean())
        print(f"  {cls:8s} {det[m].mean() * 100:5.1f}%")
    import json
    stats = dict(n_images=int(len(det)), n_detected=int(det.sum()), det_rate=float(det.mean()),
                 per_class=args.per_class, workers=args.workers, images_per_s=len(items) / elapsed,
                 min_conf=args.min_conf, pad=args.pad, per_class_detection=per_class,
                 worst_detection=sorted(per_class.items(), key=lambda kv: kv[1])[:5])
    with open(args.out.replace(".npz", "") + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
