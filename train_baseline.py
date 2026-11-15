"""
Baseline and iteration experiments for static ASL fingerspelling on cached
MediaPipe landmarks.

Experiments (each is train -> evaluate on a held-out frame range):

  baseline    raw 63-dim MediaPipe coordinates  + logistic regression
  iter1       wrist-centered, scale-normalized  + logistic regression
  iter2       normalized + pairwise distances   + MLP

Split: the Kaggle images are consecutive frames of one signer, so we hold out
the LAST 20% of frame indices of every class (a contiguous block) rather than a
random 20%. That keeps near-duplicate neighbouring frames out of the test set.

Outputs (in --out, default results/):
  results.json                metrics per experiment
  results.md                  markdown table for the report
  confusion_<exp>.png         confusion matrix
  model_<exp>.joblib          fitted pipeline, used by the demo
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

EXPERIMENTS = {
    "baseline": dict(view="raw", model="logreg"),
    "iter1": dict(view="normalized", model="logreg"),
    "iter2": dict(view="normalized+dist", model="mlp"),
}


def make_model(kind, seed):
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                            early_stopping=True, random_state=seed)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def plot_confusion(cm, classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train_landmarks.npz")
    ap.add_argument("--test-cache", default=None, help="optional extra test set (flat Kaggle test folder)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["signer", "block"], default="signer",
                    help="signer: hold out one person (needs 'signer' in the cache); block: last 20%% of frames per class")
    ap.add_argument("--test-signer", default="hand2")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.cache, allow_pickle=False)
    lm, det, hands, y, idx = d["landmarks"], d["detected"], d["handedness"], d["label"], d["index"]
    classes = sorted(set(y))
    print(f"{len(y)} samples, {len(classes)} classes, detection rate {det.mean() * 100:.1f}%")

    signer = d["signer"] if "signer" in d.files else np.array([""] * len(y))
    if args.split == "signer" and (signer == args.test_signer).any():
        # Signer-independent: every image of one person is held out.
        test_mask = signer == args.test_signer
        print(f"signer split: test on {args.test_signer}, train on {sorted(set(signer[~test_mask]))}")
    else:
        # Contiguous block split per class, on frame index.
        test_mask = np.zeros(len(y), bool)
        for c in classes:
            m = np.where(y == c)[0]
            order = m[np.argsort(idx[m])]
            n_test = int(round(len(order) * args.test_frac))
            test_mask[order[-n_test:]] = True
        print("block split: last 20% of frames per class")
    train_mask = ~test_mask

    n_test_total = test_mask.sum()
    tr = train_mask & det
    te = test_mask & det
    print(f"train: {tr.sum()} detected / {train_mask.sum()}   test: {te.sum()} detected / {n_test_total}")

    extra = None
    if args.test_cache and os.path.exists(args.test_cache):
        e = np.load(args.test_cache, allow_pickle=False)
        extra = e

    results = {}
    for name, cfg in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        X = build_features(lm, hands, cfg["view"])
        model = make_model(cfg["model"], args.seed)
        t0 = time.time()
        model.fit(X[tr], y[tr])
        fit_s = time.time() - t0

        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])
        acc = accuracy_score(y[te], pred)
        top3_idx = np.argsort(-proba, axis=1)[:, :3]
        top3 = float(np.mean([y[te][i] in model.classes_[top3_idx[i]] for i in range(len(pred))]))
        # Some sign pairs share one handshape in ASL (digit 0 / letter O, 2 / V, 6 / W).
        # 'merged' accuracy counts those as correct, an upper bound on what a
        # single-frame classifier can resolve.
        merge = {"0": "o", "2": "v", "6": "w"}
        mg = lambda a: np.array([merge.get(t, t) for t in a])
        acc_merged = float((mg(pred) == mg(y[te])).mean())
        # Strict accuracy: a frame with no detected hand counts as wrong.
        acc_strict = (pred == y[te]).sum() / n_test_total
        cm = confusion_matrix(y[te], pred, labels=classes)
        per_class = {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)}
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
        # Most confused pairs
        off = cm.copy(); np.fill_diagonal(off, 0)
        pairs = []
        for _ in range(5):
            i, j = np.unravel_index(off.argmax(), off.shape)
            if off[i, j] == 0:
                break
            pairs.append((classes[i], classes[j], int(off[i, j])))
            off[i, j] = 0

        r = dict(view=cfg["view"], model=cfg["model"], n_features=int(X.shape[1]), split=args.split,
                 detection_rate=float(det.mean()),
                 n_train=int(tr.sum()), n_test=int(te.sum()), n_test_total=int(n_test_total),
                 accuracy=float(acc), top3=float(top3), accuracy_strict=float(acc_strict), accuracy_merged=acc_merged,
                 fit_seconds=fit_s, worst_classes=worst, confused_pairs=pairs)
        if extra is not None:
            Xe = build_features(extra["landmarks"], extra["handedness"], cfg["view"])
            de = extra["detected"]
            pe = model.predict(Xe[de])
            r["kaggle_test_acc"] = float((pe == extra["label"][de]).mean())
            r["kaggle_test_n"] = int(de.sum())
        results[name] = r
        plot_confusion(cm, classes, f"{name}: {cfg['view']} + {cfg['model']}  (acc {acc * 100:.1f}%)",
                       os.path.join(args.out, f"confusion_{name}.png"))
        joblib.dump(dict(model=model, view=cfg["view"], classes=list(model.classes_)),
                    os.path.join(args.out, f"model_{name}.joblib"))
        print(f"\n[{name}] {cfg['view']} + {cfg['model']}  features={X.shape[1]}")
        print(f"  accuracy {acc * 100:.2f}%   top-3 {top3 * 100:.2f}%   strict {acc_strict * 100:.2f}%   merged {acc_merged * 100:.2f}%   fit {fit_s:.1f}s")
        print(f"  worst classes: {', '.join(f'{c} {a * 100:.0f}%' for c, a in worst)}")
        print(f"  confused pairs (true->pred): {', '.join(f'{a}->{b} x{n}' for a, b, n in pairs)}")
        if "kaggle_test_acc" in r:
            print(f"  kaggle 28-image test set: {r['kaggle_test_acc'] * 100:.1f}% ({r['kaggle_test_n']} detected)")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["| Experiment | Features | Model | Dim | Test acc | Top-3 | Strict acc | Merged acc |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['view']} | {r['model']} | {r['n_features']} | "
                     f"{r['accuracy'] * 100:.1f}% | {r['top3'] * 100:.1f}% | {r['accuracy_strict'] * 100:.1f}% | {r['accuracy_merged'] * 100:.1f}% |")
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
