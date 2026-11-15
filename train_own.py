"""
Iteration 5: our own recordings (real background, real webcam).

Combines the Kaggle cache with every data_own/*.npz and runs the transfer
experiments that matter for a new user:

  kaggle -> <signer>            train on Kaggle only, test on each of us
  kaggle+aug -> <signer>        same, with tilt+jitter augmentation
  kaggle+other -> <signer>      add the OTHER person's recordings to training
  other -> <signer>             train on the other person only (no Kaggle)
  kaggle+other+aug -> <signer>  everything

Every test is on frames the model has never seen from a person it has never
seen. Also reports the Kaggle hand2 number for the kaggle+own models so we can
see whether our data helps or hurts there.

  python train_own.py                       # uses cache/train_landmarks.npz + data_own/*.npz
  python train_own.py --copies 5 --model logreg
Outputs: results_own/results.json, results.md, transfer_bar.png, model_final.joblib
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment import augment
from features import build_features

MERGE = {"0": "o", "2": "v", "6": "w"}
VIEW = "normalized"


def load_all(cache, own_glob):
    parts = []
    d = np.load(cache, allow_pickle=False)
    parts.append(dict(lm=d["landmarks"], det=d["detected"], hands=d["handedness"], y=d["label"],
                      signer=d["signer"], source=np.array(["kaggle"] * len(d["label"]))))
    for p in sorted(glob.glob(own_glob)):
        d = np.load(p, allow_pickle=False)
        parts.append(dict(lm=d["landmarks"], det=d["detected"], hands=d["handedness"], y=d["label"],
                          signer=d["signer"], source=np.array(["own"] * len(d["label"]))))
        print(f"loaded {p}: {len(d['label'])} frames, {d['detected'].mean() * 100:.0f}% with a hand, "
              f"signer {set(d['signer'])}, {len(set(d['label']))} signs")
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def make_model(kind, seed):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                                         early_stopping=True, random_state=seed))


def run(D, train_mask, test_mask, aug, copies, kind, seed, drop_motion=("j", "z")):
    rng = np.random.default_rng(seed)
    tr = train_mask & D["det"]
    te = test_mask & D["det"]
    if drop_motion:  # J and Z are motion letters; skip them for the static classifier
        keep = ~np.isin(D["y"], drop_motion)
        tr, te = tr & keep, te & keep
    lm, y, h = D["lm"][tr], D["y"][tr], D["hands"][tr]
    if aug:
        lm, y, h = augment(lm, y, h, rng, copies=copies, ops=("tilt", "jitter"))
    model = make_model(kind, seed).fit(build_features(lm, h, VIEW), y)
    pred = model.predict(build_features(D["lm"][te], D["hands"][te], VIEW))
    yt = D["y"][te]
    mg = lambda a: np.array([MERGE.get(t, t) for t in a])
    classes = sorted(set(yt))
    per = {c: float((pred[yt == c] == c).mean()) for c in classes}
    return dict(accuracy=float((pred == yt).mean()),
                strict=float((pred == yt).sum() / max(1, (test_mask & keep).sum() if drop_motion else test_mask.sum())),
                merged=float((mg(pred) == mg(yt)).mean()), n_train=int(len(y)), n_test=int(te.sum()),
                worst_classes=sorted(per.items(), key=lambda kv: kv[1])[:6]), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train_landmarks.npz")
    ap.add_argument("--own", default="data_own/*.npz")
    ap.add_argument("--out", default="results_own")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = load_all(args.cache, args.own)
    own_signers = sorted(set(D["signer"][D["source"] == "own"]))
    if not own_signers:
        raise SystemExit("no data_own/*.npz files found, record first with record_signs.py")
    kaggle_train = (D["source"] == "kaggle") & (D["signer"] != "hand2")
    kaggle_test = (D["source"] == "kaggle") & (D["signer"] == "hand2")

    results = {}
    best = None
    for s in own_signers:
        me = D["signer"] == s
        others = (D["source"] == "own") & ~me
        exps = {
            f"kaggle -> {s}": (kaggle_train, False),
            f"kaggle+aug -> {s}": (kaggle_train, True),
        }
        if others.any():
            exps[f"other -> {s}"] = (others, False)
            exps[f"kaggle+other -> {s}"] = (kaggle_train | others, False)
            exps[f"kaggle+other+aug -> {s}"] = (kaggle_train | others, True)
        for name, (trm, aug) in exps.items():
            accs = []
            for seed in range(args.seeds):
                r, model = run(D, trm, me, aug, args.copies, args.model, seed)
                accs.append(r["accuracy"])
                if seed == 0:
                    r0, m0 = r, model
                    # how does the same model do on Kaggle's held-out person?
                    rk, _ = run(D, trm, kaggle_test, aug, args.copies, args.model, seed)
                    r0["kaggle_hand2_accuracy"] = rk["accuracy"]
            r0["accuracy"], r0["accuracy_std"] = float(np.mean(accs)), float(np.std(accs))
            results[name] = r0
            print(f"[{name:28s}] train {r0['n_train']:6d}  test {r0['n_test']:5d}  acc {r0['accuracy'] * 100:5.1f} "
                  f"+/- {r0['accuracy_std'] * 100:.1f}  merged {r0['merged'] * 100:5.1f}  hand2 {r0['kaggle_hand2_accuracy'] * 100:5.1f}"
                  f"   worst: {', '.join(f'{c.upper()} {a * 100:.0f}%' for c, a in r0['worst_classes'][:4])}")
            if name.startswith("kaggle+other+aug"):
                best = m0

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["| Train -> test | Train size | Test frames | Test acc | Merged | Kaggle hand2 |", "|---|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['n_train']:,} | {r['n_test']:,} | {r['accuracy'] * 100:.1f} +/- {r['accuracy_std'] * 100:.1f}% | "
                     f"{r['merged'] * 100:.1f}% | {r['kaggle_hand2_accuracy'] * 100:.1f}% |")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    if best is not None:
        # final model for the demo: trained on Kaggle + everyone's recordings + augmentation
        allm = (D["source"] == "kaggle") | (D["source"] == "own")
        r, model = run(D, allm, kaggle_test, True, args.copies, args.model, 0)
        joblib.dump(dict(model=model, view=VIEW, classes=list(model.classes_)), os.path.join(args.out, "model_final.joblib"))
        print(f"saved {args.out}/model_final.joblib (trained on everything, {r['n_train']:,} samples)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(results)
    ax.bar(range(len(names)), [results[n]["accuracy"] * 100 for n in names],
           yerr=[results[n]["accuracy_std"] * 100 for n in names], capsize=3, color="#4c78a8")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("accuracy on held-out person (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Iteration 5: transfer to our own webcam recordings")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "transfer_bar.png"), dpi=170)


if __name__ == "__main__":
    main()
