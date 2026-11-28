"""
Iteration 9: single pass vs two-stage landmarks on real webcam data, and the final
letter model for the demo.

eva_close.npz (v2 recorder) stores two landmark sets from the SAME frames:
  landmarks_full  one MediaPipe pass on the full 1080p frame
  landmarks       two-stage: detect, crop around the hand, run MediaPipe again
So the only thing that changes between the two rows is the landmark extractor.

Comparison (letter accuracy on eva_close, J/Z excluded, digits merged as in letters.py):
  unseen signer   train on Kaggle + Isha's recording, test on Eva
  new session     train on Kaggle + Isha + Eva's older room recording, test on eva_close
Both are reported overall and by hand size in the frame (simulated small hands are
in results_twostage/; this is the real-data check).

Final model: Kaggle + every data_own recording (two-stage where available), tilt+jitter
augmentation, logistic regression on normalized xy. Saved for demo_predict.py.

  python train_final.py
Outputs: results_final/results.md, results.json, hand_size_acc.png, model_final.joblib
"""

import argparse
import json
import os

import joblib
import numpy as np

from letters import VIEW, load_letters, predict_proba, train_letters

BINS = [0.0, 0.45, 0.55, 0.65, 9.0]
BIN_NAMES = ["< 0.45", "0.45-0.55", "0.55-0.65", ">= 0.65"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-rec", default="eva_close")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="results_final")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    full = np.load(f"data_own/{args.test_rec}.npz", allow_pickle=False)
    pts = full["landmarks_full"][..., :2]
    # hand size = longer side of the hand's bounding box, in frame-height units (single pass, same for both
    # rows). Height alone is misleading: sideways letters (G, H, P, Q) are short but not small.
    hand_h = (pts.max(1) - pts.min(1)).max(1)

    results = {}
    for key, name in [("landmarks_full", "single pass"), ("landmarks", "two-stage")]:
        D = load_letters(key=key)
        test = (D["rec"] == args.test_rec) & D["keep"]
        test_signer = D["signer"][D["rec"] == args.test_rec][0]
        h = np.zeros(len(D["rec"]))
        h[D["rec"] == args.test_rec] = hand_h
        setups = {"unseen signer": D["signer"] != test_signer, "new session": D["rec"] != args.test_rec}
        for setup, train_mask in setups.items():
            accs, bins = [], []
            for seed in range(args.seeds):
                model, n_train = train_letters(D, train_mask, seed=seed)
                pred = model.classes_[predict_proba(model, D["lm"][test], D["hands"][test]).argmax(1)]
                ok = pred == D["y"][test]
                accs.append(ok.mean())
                if seed == 0 and setup == "unseen signer" and key == "landmarks":
                    pairs = {}
                    for t, q in zip(D["y"][test], pred):
                        if t != q:
                            k = "".join(sorted((t.upper(), q.upper())))
                            pairs[k] = pairs.get(k, 0) + 1
                    confusions = sorted(pairs.items(), key=lambda kv: -kv[1])[:15]
                b = np.digitize(h[test], BINS[1:-1])
                bins.append([float(ok[b == i].mean()) if (b == i).any() else float("nan") for i in range(len(BIN_NAMES))])
                counts = [int((b == i).sum()) for i in range(len(BIN_NAMES))]
            results[f"{setup} | {name}"] = dict(setup=setup, landmarks=name, accuracy=float(np.mean(accs)),
                                                accuracy_std=float(np.std(accs)), n_train=n_train,
                                                n_test=int(test.sum()), by_hand_height=np.nanmean(bins, 0).tolist(),
                                                bin_counts=counts)
            r = results[f"{setup} | {name}"]
            print(f"[{setup:13s} | {name:11s}] acc {r['accuracy'] * 100:5.1f} +/- {r['accuracy_std'] * 100:.1f}%  "
                  f"by hand size " + "  ".join(f"{n} {a * 100:.0f}%" for n, a in zip(BIN_NAMES, r["by_hand_height"])))

    # final model on everything, two-stage landmarks
    D = load_letters(key="landmarks")
    model, n_train = train_letters(D, np.ones(len(D["y"]), bool), seed=0)
    joblib.dump(dict(model=model, view=VIEW, classes=list(model.classes_),
                     trained_on=sorted(set(D["rec"].tolist())), n_train=n_train),
                os.path.join(args.out, "model_final.joblib"))
    print(f"saved {args.out}/model_final.joblib ({n_train:,} samples incl. augmentation, "
          f"{len(model.classes_)} letters, recordings {sorted(set(D['rec'].tolist()))})")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(test=args.test_rec, bins=BIN_NAMES, rows=results, final_n_train=n_train,
                       confusable_pairs=confusions), f, indent=2)

    L = [f"## Iteration 9: single pass vs two-stage on {args.test_rec} (same frames)", "",
         f"Letter accuracy (J/Z excluded, 24 letters). Hand size = longer side of the hand's box as a fraction of the frame height. {args.seeds} seeds.", "",
         "| Train | Landmarks | Test acc | " + " | ".join(f"size {n}" for n in BIN_NAMES) + " |",
         "|---|---|---|" + "---|" * len(BIN_NAMES)]
    for r in results.values():
        L.append(f"| {r['setup']} | {r['landmarks']} | {r['accuracy'] * 100:.1f} +/- {r['accuracy_std'] * 100:.1f}% | " +
                 " | ".join(f"{a * 100:.1f}%" for a in r["by_hand_height"]) + " |")
    any_r = next(iter(results.values()))
    L += ["", "Frames per hand-size bin: " + ", ".join(f"{n}: {c}" for n, c in zip(BIN_NAMES, any_r["bin_counts"])),
          "", "unseen signer = Kaggle + Isha's recording (no Eva data). new session = also Eva's older room recording.",
          "", "Most confused letter pairs (unseen signer, two-stage; used by eval_cleanup.py): " +
          ", ".join(f"{p[0]}/{p[1]} ({c})" for p, c in confusions[:10]),
          "", f"Final demo model: {args.out}/model_final.joblib, trained on Kaggle + all of data_own "
              f"({n_train:,} samples after augmentation), two-stage landmarks."]
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(BIN_NAMES))
    for i, (name, color) in enumerate([("single pass", "#9aa5b1"), ("two-stage", "#4c78a8")]):
        r = results[f"unseen signer | {name}"]
        ax.bar(x + (i - 0.5) * 0.38, [a * 100 for a in r["by_hand_height"]], 0.38, label=name, color=color)
    ax.set_xticks(x, [f"{n}\n(n={c})" for n, c in zip(BIN_NAMES, any_r["bin_counts"])])
    ax.set_xlabel("hand size (longer side of hand box) / frame height")
    ax.set_ylabel("letter accuracy on Eva (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.set_title("Iteration 9: single pass vs two-stage, unseen signer")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "hand_size_acc.png"), dpi=170)


if __name__ == "__main__":
    main()
