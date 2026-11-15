"""
Iteration 4: landmark augmentation.

Question: does synthesizing extra training hands (rotated, scaled, tilted,
jittered) help the classifier generalize to a person it has never seen?

Starting point is iteration 2 from the proposal: padded extraction, normalized
landmarks, logistic regression, signer-independent test on hand2.

Experiments (each run with several seeds, mean +/- std reported):
  none            no augmentation (= proposal iteration 2)
  rotate          in-plane rotation only
  scale           scale only (a no-op after normalization, kept as a control)
  tilt            3D tilt about the vertical axis only
  jitter          per-point Gaussian noise only
  all             rotate + scale + tilt + jitter
  rotnorm         no augmentation, but rotation-normalized features (the
                  "fix it in features" alternative to rotation augmentation)
  all+rotnorm     both
Each of the above is run with logistic regression and with the MLP.

Outputs in --out (default results_aug/): results.json, results.md,
augmentation_bar.png, model_best.joblib.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment import augment
from features import build_features

MERGE = {"0": "o", "2": "v", "6": "w"}
EXPERIMENTS = {
    "none": dict(ops=(), view="normalized"),
    "rotate": dict(ops=("rotate",), view="normalized"),
    "scale": dict(ops=("scale",), view="normalized"),
    "tilt": dict(ops=("tilt",), view="normalized"),
    "jitter": dict(ops=("jitter",), view="normalized"),
    "tilt+jitter": dict(ops=("tilt", "jitter"), view="normalized"),
    "all": dict(ops=("rotate", "scale", "tilt", "jitter"), view="normalized"),
    "rotnorm": dict(ops=(), view="normalized+rot"),
    "all+rotnorm": dict(ops=("rotate", "scale", "tilt", "jitter"), view="normalized+rot"),
}


def make_model(kind, seed):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                                         early_stopping=True, random_state=seed))


def evaluate(model, X, y, n_total):
    pred = model.predict(X)
    acc = float((pred == y).mean())
    strict = float((pred == y).sum() / n_total)
    mg = lambda a: np.array([MERGE.get(t, t) for t in a])
    merged = float((mg(pred) == mg(y)).mean())
    return acc, strict, merged, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train_landmarks.npz")
    ap.add_argument("--out", default="results_aug")
    ap.add_argument("--test-signer", default="hand2")
    ap.add_argument("--copies", type=int, default=5, help="augmented copies per training sample")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=["logreg", "mlp"])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.cache, allow_pickle=False)
    lm, det, hands, y, signer = d["landmarks"], d["detected"], d["handedness"], d["label"], d["signer"]
    test = signer == args.test_signer
    tr, te = (~test) & det, test & det
    n_test_total = int(test.sum())
    print(f"train {tr.sum()} / test {te.sum()} detected of {n_test_total} ({args.test_signer})")

    results = {}
    for name, cfg in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        for mk in args.models:
            key = f"{name}/{mk}"
            accs, stricts, mergeds, times = [], [], [], []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                if cfg["ops"]:
                    la, ya, ha = augment(lm[tr], y[tr], hands[tr], rng, copies=args.copies, ops=cfg["ops"])
                else:
                    la, ya, ha = lm[tr], y[tr], hands[tr]
                Xtr = build_features(la, ha, cfg["view"])
                Xte = build_features(lm[te], hands[te], cfg["view"])
                t0 = time.time()
                model = make_model(mk, seed).fit(Xtr, ya)
                times.append(time.time() - t0)
                acc, strict, merged, pred = evaluate(model, Xte, y[te], n_test_total)
                accs.append(acc); stricts.append(strict); mergeds.append(merged)
                if seed == 0:
                    first_model, first_pred = model, pred
            r = dict(experiment=name, model=mk, ops=list(cfg["ops"]), view=cfg["view"],
                     n_train=int(len(ya)), n_test=int(te.sum()),
                     accuracy=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
                     strict=float(np.mean(stricts)), merged=float(np.mean(mergeds)),
                     fit_seconds=float(np.mean(times)), seeds=args.seeds)
            # worst classes and confused pairs from the first seed
            classes = sorted(set(y[te]))
            per = {c: float((first_pred[y[te] == c] == c).mean()) for c in classes}
            r["worst_classes"] = sorted(per.items(), key=lambda kv: kv[1])[:5]
            wrong = [(a, b) for a, b in zip(y[te], first_pred) if a != b]
            from collections import Counter
            r["confused_pairs"] = [(a, b, n) for (a, b), n in Counter(wrong).most_common(5)]
            results[key] = r
            joblib.dump(dict(model=first_model, view=cfg["view"], classes=list(first_model.classes_)),
                        os.path.join(args.out, f"model_{name.replace('+', '_')}_{mk}.joblib"))
            print(f"[{key:20s}] n_train={len(ya):6d}  acc {np.mean(accs) * 100:5.1f} +/- {np.std(accs) * 100:.1f}"
                  f"  strict {np.mean(stricts) * 100:5.1f}  merged {np.mean(mergeds) * 100:5.1f}  fit {np.mean(times):.1f}s")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["| Augmentation | Features | Model | Train size | Test acc | Strict | Merged |", "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {r['experiment']} | {r['view']} | {r['model']} | {r['n_train']:,} | "
                     f"{r['accuracy'] * 100:.1f} +/- {r['accuracy_std'] * 100:.1f}% | {r['strict'] * 100:.1f}% | {r['merged'] * 100:.1f}% |")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in EXPERIMENTS if (not args.only or n in args.only)]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    for i, mk in enumerate(args.models):
        vals = [results[f"{n}/{mk}"]["accuracy"] * 100 for n in names]
        errs = [results[f"{n}/{mk}"]["accuracy_std"] * 100 for n in names]
        ax.bar(np.arange(len(names)) + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=mk)
    base = results.get("none/logreg", {}).get("accuracy", 0) * 100
    ax.axhline(base, color="gray", ls="--", lw=1, label=f"no augmentation, logreg ({base:.1f}%)")
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_ylabel("accuracy on held-out signer (%)")
    ax.set_ylim(max(0, min(r["accuracy"] for r in results.values()) * 100 - 5), 100)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Iteration 4: landmark augmentation, test on hand2")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "augmentation_bar.png"), dpi=170)


if __name__ == "__main__":
    main()
