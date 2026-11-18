"""
Iteration 6 experiment: does a second MediaPipe pass on a crop fix small hands?

We take the held-out Kaggle person (hand2), paste each image at a chosen size
into a 1920x1080 black frame (like a webcam frame with the hand far away), and
compare two ways of getting landmarks:

  single   MediaPipe on the full frame (what record_signs.py v1 did)
  twostage MediaPipe on the full frame, crop around the hand with margin,
           MediaPipe again on the crop (twostage.py)

Both are classified by the iteration 2 model trained on the other four people.
Hand height fraction 0.83 is the original image size, 0.3 and 0.2 are what our
first webcam recordings looked like.

  python sim_small_hand.py --images data2/asl_dataset --out results_twostage
"""

import argparse
import glob
import json
import os
import re

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features, to_square_coords
from twostage import TwoStage, full_frame_points

W, H = 1920, 1080


def paste(img, frac, rng):
    """Put the (square) image into a black WxH frame so its height is frac*H, at a random position."""
    side = int(frac * H)
    small = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    frame = np.zeros((H, W, 3), np.uint8)
    x0 = rng.integers(0, W - side + 1)
    y0 = rng.integers(0, H - side + 1)
    frame[y0:y0 + side, x0:x0 + side] = small
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data2/asl_dataset")
    ap.add_argument("--signer", default="hand2")
    ap.add_argument("--cache", default="cache/train_landmarks.npz")
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.83, 0.5, 0.35, 0.25, 0.2])
    ap.add_argument("--per-class", type=int, default=10, help="images per class per size (speed)")
    ap.add_argument("--out", default="results_twostage")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    # classifier: iteration 2 (normalized landmarks + logistic regression) on the other four people
    k = np.load(args.cache, allow_pickle=False)
    tr = (k["signer"] != args.signer) & k["detected"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(
        build_features(k["landmarks"][tr], k["handedness"][tr], "normalized"), k["label"][tr])

    # test images
    seen, items = set(), []
    for p in sorted(glob.glob(os.path.join(args.images, "**", f"{args.signer}_*"), recursive=True)):
        b = os.path.basename(p)
        if b in seen or not b.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        seen.add(b)
        items.append((p, re.match(rf"{args.signer}_([a-z0-9])_", b).group(1)))
    by_cls = {}
    for p, c in items:
        by_cls.setdefault(c, []).append(p)
    # sample across camera angles (file names: bot, dif, left, right, top), not the first N
    items = [(p, c) for c in sorted(by_cls)
             for p in rng.choice(by_cls[c], size=min(args.per_class, len(by_cls[c])), replace=False)]
    print(f"{len(items)} test images, {len(by_cls)} classes")

    results = {}
    with TwoStage(args.model) as ts:
        for frac in args.fracs:
            rows = {"single": [], "twostage": []}
            det = {"single": 0, "twostage": 0}
            ys = []
            for p, c in items:
                frame = paste(cv2.imread(p), frac, rng)
                r = ts.detect_image(frame)
                pts1, hand1 = full_frame_points(r)
                pts2, hand2, _ = ts.refine(frame, r)
                ys.append(c)
                for name, pts, hand in (("single", pts1, hand1), ("twostage", pts2, hand2)):
                    if pts is None:
                        rows[name].append(None)
                    else:
                        det[name] += 1
                        # Kaggle images were extracted unmirrored, same as this canvas, so no label swap
                        rows[name].append((to_square_coords(pts, W, H), hand))
            ys = np.array(ys)
            res = {}
            for name in rows:
                ok = np.array([x is not None for x in rows[name]])
                if ok.sum() == 0:
                    res[name] = dict(detection=0.0, accuracy=0.0, strict=0.0)
                    continue
                lm = np.stack([x[0] for x in rows[name] if x is not None])
                hd = np.array([x[1] for x in rows[name] if x is not None])
                pred = clf.predict(build_features(lm, hd, "normalized"))
                acc = float((pred == ys[ok]).mean())
                res[name] = dict(detection=float(ok.mean()), accuracy=acc, strict=float((pred == ys[ok]).sum() / len(ys)))
            results[f"{frac:.2f}"] = res
            print(f"hand height {frac:.2f} of frame:  single  det {res['single']['detection'] * 100:5.1f}%  acc {res['single']['accuracy'] * 100:5.1f}%  "
                  f"| twostage  det {res['twostage']['detection'] * 100:5.1f}%  acc {res['twostage']['accuracy'] * 100:5.1f}%", flush=True)

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    lines = ["| Hand height / frame | Single pass: found | Single: acc | Two-stage: found | Two-stage: acc |", "|---|---|---|---|---|"]
    for f, r in results.items():
        lines.append(f"| {f} | {r['single']['detection'] * 100:.0f}% | {r['single']['accuracy'] * 100:.1f}% | "
                     f"{r['twostage']['detection'] * 100:.0f}% | {r['twostage']['accuracy'] * 100:.1f}% |")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = [float(f) for f in results]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(fr, [results[f]["single"]["accuracy"] * 100 for f in results], "o-", label="single pass (full frame)")
    ax.plot(fr, [results[f]["twostage"]["accuracy"] * 100 for f in results], "s-", label="two-stage (crop and re-detect)")
    ax.set_xlabel("hand height as a fraction of the frame")
    ax.set_ylabel("accuracy on hand2 (%)")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Iteration 6: small hands in a 1920x1080 frame")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "small_hand.png"), dpi=170)


if __name__ == "__main__":
    main()
