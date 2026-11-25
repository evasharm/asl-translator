"""
Iteration 7: word-level signs from the Google / Kaggle "ASL Signs" landmarks.

Input is the subset made by prep_signs.py (every clip resampled to T frames,
both hands + 6 arm points per frame). Two models are compared on participants
the model has never seen (signer-independent, like hand2 for the letters):

  baseline   frame-average: mean / std / min / max of each per-frame feature
             over the clip, then logistic regression. Ignores the ORDER of
             frames, so it shows how much the sequence model gains from motion.
  gru        2-layer bidirectional GRU over the T frames, mean+max pooled.

Per-frame features (all xy only, z dropped like the letter model):
  shape     each hand wrist-centered and scaled by wrist->middle-knuckle length
            (the same normalization as features.py), 2 x 42 values
  position  dominant wrist relative to its average position in the clip, and
            the other wrist relative to the dominant wrist, in hand-size units.
            Needs no pose, so the live demo can compute it from MediaPipe Hands.
  present   1/0 flag per hand (missing hands are zero-filled)
  --arms    add shoulders/elbows/wrists from pose, relative to the shoulder
            midpoint in shoulder-width units (ablation: does the body help?)
  --velocity  append frame-to-frame differences of everything above

Each clip is mirrored if the signer used the left hand more, so every clip
looks right-handed (same idea as mirroring left hands for the letters).

  pip install torch --index-url https://download.pytorch.org/whl/cpu
  python train_signs.py --data cache/asl_signs_20.npz --out results_signs_20
  python train_signs.py --data cache/asl_signs_20.npz --out results_signs_20 --grid   # full ablation

Outputs in --out: results.json, results.md, model_gru.pt (best config, for demo_predict.py)
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WRIST, MIDDLE_MCP = 0, 9
# pose points saved by prep_signs.py: 11 L shoulder, 12 R shoulder, 13 L elbow, 14 R elbow, 15 L wrist, 16 R wrist
POSE_SWAP = [1, 0, 3, 2, 5, 4]


# ----------------------------------------------------------------------------- features

def canonicalize(hands, pose):
    """Mirror clips signed mainly with the left hand so the dominant hand is always slot 1.
    hands: (N, T, 2, 21, 3), pose: (N, T, 6, 3) or None. Returns copies."""
    hands = hands.astype(np.float32).copy()
    pose = None if pose is None else pose.astype(np.float32).copy()
    vis = ~np.isnan(hands[:, :, :, WRIST, 0])            # (N, T, 2)
    left_dom = vis[:, :, 0].sum(1) > vis[:, :, 1].sum(1)  # (N,)
    if left_dom.any():
        h = hands[left_dom][:, :, ::-1].copy()            # swap the two hand slots
        h[..., 0] = 1.0 - h[..., 0]                       # mirror x
        hands[left_dom] = h
        if pose is not None:
            p = pose[left_dom][:, :, POSE_SWAP].copy()
            p[..., 0] = 1.0 - p[..., 0]
            pose[left_dom] = p
    return hands, pose, left_dom


def clip_features(hands, pose=None, arms=False, velocity=True):
    """hands: (N, T, 2, 21, 3) already canonicalized (slot 1 = dominant). Returns (N, T, F) float32."""
    xy = hands[..., :2]                                   # (N, T, 2, 21, 2)
    present = ~np.isnan(xy[..., WRIST, 0])                # (N, T, 2)
    wrist = xy[..., WRIST:WRIST + 1, :]
    size = np.linalg.norm(xy[..., MIDDLE_MCP, :] - xy[..., WRIST, :], axis=-1)  # (N, T, 2)
    size = np.where(np.isnan(size) | (size < 1e-4), np.nan, size)
    shape = (xy - wrist) / size[..., None, None]
    shape = np.nan_to_num(shape).reshape(*xy.shape[:2], -1)         # (N, T, 84)

    # one hand-size unit per clip: median dominant-hand size over visible frames
    with np.errstate(all="ignore"):
        unit = np.nanmedian(np.where(present, size, np.nan).reshape(len(xy), -1), axis=1)  # (N,)
    unit = np.where(np.isnan(unit), 0.1, unit)[:, None, None]
    dom = xy[:, :, 1, WRIST]                              # (N, T, 2)
    other = xy[:, :, 0, WRIST]
    with np.errstate(all="ignore"):
        dom_mean = np.nanmean(dom, axis=1, keepdims=True)
    pos_dom = np.nan_to_num((dom - dom_mean) / unit)
    pos_other = np.nan_to_num((other - dom) / unit)

    parts = [shape, pos_dom, pos_other, present.astype(np.float32)]
    if arms and pose is not None:
        pxy = pose[..., :2]                               # (N, T, 6, 2)
        mid = (pxy[:, :, 0] + pxy[:, :, 1]) / 2
        width = np.linalg.norm(pxy[:, :, 0] - pxy[:, :, 1], axis=-1)
        width = np.where(np.isnan(width) | (width < 1e-4), np.nan, width)
        parts.append(np.nan_to_num((pxy - mid[:, :, None]) / width[..., None, None]).reshape(*pxy.shape[:2], -1))
    f = np.concatenate(parts, axis=-1).astype(np.float32)
    if velocity:
        v = np.diff(f, axis=1, prepend=f[:, :1])
        f = np.concatenate([f, v], axis=-1)
    return np.clip(f, -20, 20)


def augment_clips(hands, pose, rng):
    """Random scale, small in-plane rotation and time crop, per clip. Works on raw coordinates."""
    N, T = hands.shape[:2]
    hands, pose = hands.copy(), None if pose is None else pose.copy()
    for i in range(N):
        s = rng.uniform(0.85, 1.15)
        a = np.deg2rad(rng.uniform(-12, 12))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32) * s
        c = np.array([0.5, 0.5], np.float32)
        hands[i, ..., :2] = (hands[i, ..., :2] - c) @ R.T + c
        if pose is not None:
            pose[i, ..., :2] = (pose[i, ..., :2] - c) @ R.T + c
        # time crop: keep a random 80-100% window, stretch it back to T frames
        keep = int(T * rng.uniform(0.8, 1.0))
        start = rng.integers(0, T - keep + 1)
        idx = np.round(np.linspace(start, start + keep - 1, T)).astype(int)
        hands[i] = hands[i, idx]
        if pose is not None:
            pose[i] = pose[i, idx]
    return hands, pose


# ----------------------------------------------------------------------------- split

def participant_split(participant, test_frac=0.2, seed=0):
    """Hold out whole participants. Returns (train_mask, val_mask, test_mask)."""
    ps = np.array(sorted(set(participant.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(ps)
    n_test = max(1, int(round(len(ps) * test_frac)))
    test_p, val_p = ps[:n_test], ps[n_test:n_test + 1]  # one participant for early stopping
    te = np.isin(participant, test_p)
    va = np.isin(participant, val_p)
    return ~te & ~va, va, te


# ----------------------------------------------------------------------------- models

def topk(proba, y, classes, k=5):
    order = np.argsort(-proba, axis=1)[:, :k]
    yi = np.searchsorted(classes, y)
    return float((order == yi[:, None]).any(1).mean())


def run_baseline(F, y, tr, te):
    def pool(x):
        return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    model.fit(pool(F[tr]), y[tr])
    proba = model.predict_proba(pool(F[te]))
    pred = model.classes_[proba.argmax(1)]
    return dict(accuracy=float((pred == y[te]).mean()), top5=topk(proba, y[te], model.classes_)), pred


def run_gru(hands, pose, y, classes, tr, va, te, cfg, seed, epochs, log=print):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    yi = np.searchsorted(classes, y)
    feat = lambda h, p: clip_features(h, p, arms=cfg["arms"], velocity=cfg["velocity"])
    F_va, F_te = feat(hands[va], pose[va]), feat(hands[te], pose[te])
    F_tr0 = feat(hands[tr], pose[tr])
    dim = F_tr0.shape[-1]
    mu, sd = F_tr0.reshape(-1, dim).mean(0), F_tr0.reshape(-1, dim).std(0) + 1e-6

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
            self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes)))

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1))

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * int(np.ceil(tr.sum() / 64)))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    T = lambda F: torch.from_numpy((F - mu) / sd).float()

    def predict(F):
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(T(F)), 1).numpy()

    best, best_state, best_ep = -1.0, None, 0
    h_tr, p_tr, y_tr = hands[tr], pose[tr], yi[tr]
    for ep in range(epochs):
        if cfg["augment"]:
            ha, pa = augment_clips(h_tr, p_tr, rng)
            F_tr = feat(ha, pa)
        else:
            F_tr = F_tr0
        net.train()
        perm = rng.permutation(len(y_tr))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            loss = lossf(net(T(F_tr[idx])), torch.from_numpy(y_tr[idx]).long())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = float((predict(F_va).argmax(1) == yi[va]).mean())
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 10 == 0:
            log(f"    epoch {ep + 1}/{epochs}  loss {loss.item():.3f}  val acc {acc * 100:.1f}%")
    net.load_state_dict(best_state)
    proba = predict(F_te)
    pred = classes[proba.argmax(1)]
    res = dict(accuracy=float((pred == y[te]).mean()), top5=topk(proba, y[te], classes),
               val_accuracy=best, best_epoch=best_ep + 1)
    bundle = dict(state_dict=best_state, dim=dim, mu=mu, sd=sd, classes=list(classes), cfg=cfg,
                  frames=int(hands.shape[1]))
    return res, pred, bundle


# ----------------------------------------------------------------------------- main

def worst_signs(y, pred, n=5):
    per = {c: float((pred[y == c] == c).mean()) for c in sorted(set(y))}
    return sorted(per.items(), key=lambda kv: kv[1])[:n]


def top_confusions(y, pred, n=5):
    pairs = {}
    for t, p in zip(y, pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    return [(f"{t} -> {p}", c) for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cache/asl_signs_20.npz")
    ap.add_argument("--out", default="results_signs_20")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3, help="GRU runs per config (mean +/- std reported)")
    ap.add_argument("--grid", action="store_true", help="run the full iteration 7 ablation")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of participants held out")
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    y = d["sign"].astype(str)
    participant = d["participant"]
    hands, pose, left_dom = canonicalize(d["hands"], d["pose"] if "pose" in d.files else None)
    if pose is None:
        pose = np.full(hands.shape[:2] + (6, 3), np.nan, np.float32)
    classes = np.array(sorted(set(y)))
    tr, va, te = participant_split(participant, args.test_frac, args.split_seed)
    print(f"{len(y)} clips, {len(classes)} signs, {len(set(participant.tolist()))} participants, "
          f"{hands.shape[1]} frames per clip, {left_dom.mean() * 100:.0f}% mirrored (left-hand dominant)")
    print(f"train {tr.sum()} clips / val {va.sum()} (participant {sorted(set(participant[va].tolist()))}) / "
          f"test {te.sum()} (participants {sorted(set(participant[te].tolist()))})")
    print(f"chance = {100 / len(classes):.1f}%\n")

    configs = [dict(name="gru", velocity=True, augment=True, arms=False)]
    if args.grid:
        configs = [
            dict(name="gru", velocity=False, augment=False, arms=False),
            dict(name="gru", velocity=True, augment=False, arms=False),
            dict(name="gru", velocity=True, augment=True, arms=False),
            dict(name="gru", velocity=True, augment=True, arms=True),
        ]

    rows, best = [], None
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    F = clip_features(hands, pose, arms=False, velocity=True)
    res, pred = run_baseline(F, y, tr, te)
    print(f"baseline (frame-average + logreg): {res['accuracy'] * 100:.1f}%  top-5 {res['top5'] * 100:.1f}%  "
          f"({time.time() - t0:.0f}s)")
    rows.append(dict(model="frame-average + logreg", velocity=True, augment=False, arms=False,
                     accuracy=res["accuracy"], accuracy_std=0.0, top5=res["top5"],
                     worst=worst_signs(y[te], pred), confusions=top_confusions(y[te], pred)))

    for cfg in configs:
        accs, top5s, last_pred = [], [], None
        for s in range(args.seeds):
            t0 = time.time()
            print(f"GRU velocity={cfg['velocity']} augment={cfg['augment']} arms={cfg['arms']}  seed {s}")
            res, pred, bundle = run_gru(hands, pose, y, classes, tr, va, te, cfg, s, args.epochs)
            print(f"  test {res['accuracy'] * 100:.1f}%  top-5 {res['top5'] * 100:.1f}%  "
                  f"(best val epoch {res['best_epoch']}, {time.time() - t0:.0f}s)")
            accs.append(res["accuracy"]); top5s.append(res["top5"]); last_pred = pred
            if best is None or res["val_accuracy"] > best[0]:
                best = (res["val_accuracy"], bundle)
        rows.append(dict(model="GRU", velocity=cfg["velocity"], augment=cfg["augment"], arms=cfg["arms"],
                         accuracy=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
                         top5=float(np.mean(top5s)), worst=worst_signs(y[te], last_pred),
                         confusions=top_confusions(y[te], last_pred)))

    import torch
    torch.save(best[1], os.path.join(args.out, "model_gru.pt"))

    info = dict(data=args.data, clips=int(len(y)), signs=classes.tolist(),
                participants_test=sorted(set(participant[te].tolist())),
                n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()), rows=rows)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(info, f, indent=2)

    yn = lambda b: "yes" if b else "no"
    lines = [f"Word signs: {len(classes)} signs, {len(y)} clips, test on {len(info['participants_test'])} "
             f"unseen participants ({te.sum()} clips). Chance {100 / len(classes):.1f}%.", "",
             "| Model | Velocity | Augment | Arms (pose) | Test acc | Top-5 |", "|---|---|---|---|---|---|"]
    for r in rows:
        acc = f"{r['accuracy'] * 100:.1f}%" + (f" +/- {r['accuracy_std'] * 100:.1f}" if r["model"] == "GRU" else "")
        lines.append(f"| {r['model']} | {yn(r['velocity'])} | {yn(r['augment'])} | {yn(r['arms'])} | "
                     f"{acc} | {r['top5'] * 100:.1f}% |")
    lines += ["", "Worst signs and most common confusions (last row):", ""]
    lines += [f"- {s}: {a * 100:.0f}%" for s, a in rows[-1]["worst"]]
    lines += [f"- {p}: {c} clips" for p, c in rows[-1]["confusions"]]
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {args.out}/results.md, results.json, model_gru.pt")


if __name__ == "__main__":
    main()