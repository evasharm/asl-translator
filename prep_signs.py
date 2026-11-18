"""
Make a small, shareable subset of the Google / Kaggle "ASL Signs" dataset.

The full download is ~40 GB of parquet files (one per clip, 543 landmarks per
frame: face, pose, left hand, right hand). We only need the two hands (and a
few pose points) for the most common N signs, resampled to a fixed number of
frames. That comes out to a few tens of MB, small enough to share.

  pip install pandas pyarrow
  python prep_signs.py --data data_signs --signs 20 --out cache/asl_signs_20.npz

Expected layout under --data (what `kaggle competitions download -c asl-signs`
unzips to):
  train.csv                       path, file_id, participant_id, sequence_id, sign
  train_landmark_files/<pid>/<seq>.parquet

Output .npz:
  hands        (N, T, 2, 21, 3) float16   left hand, right hand per frame; NaN = hand not visible
  pose         (N, T, 6, 3)     float16   shoulders, elbows, wrists (pose points 11-16)
  n_frames     (N,)                       original clip length
  sign         (N,)  str
  participant  (N,)  int                  for the signer-independent split
  sequence_id  (N,)  int
  signs        list of the N sign names kept, most common first
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

POSE_IDX = [11, 12, 13, 14, 15, 16]


def resample(arr, T):
    """arr: (n, ...) -> (T, ...) by linear index interpolation (NaN-aware: picks nearest frame)."""
    n = len(arr)
    if n == T:
        return arr
    idx = np.linspace(0, n - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo)[(...,) + (None,) * (arr.ndim - 1)]
    a, b = arr[lo], arr[hi]
    out = a * (1 - w) + b * w
    # where one side is NaN, take the other
    out = np.where(np.isnan(out), np.where(np.isnan(a), b, a), out)
    return out


def load_clip(path, T):
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frames = np.sort(df["frame"].unique())
    n = len(frames)
    fpos = {f: i for i, f in enumerate(frames)}
    hands = np.full((n, 2, 21, 3), np.nan, np.float32)
    pose = np.full((n, 6, 3), np.nan, np.float32)
    for side, name in enumerate(("left_hand", "right_hand")):
        d = df[df["type"] == name]
        fi = d["frame"].map(fpos).to_numpy()
        li = d["landmark_index"].to_numpy()
        hands[fi, side, li] = d[["x", "y", "z"]].to_numpy(np.float32)
    d = df[(df["type"] == "pose") & (df["landmark_index"].isin(POSE_IDX))]
    fi = d["frame"].map(fpos).to_numpy()
    li = d["landmark_index"].map({p: i for i, p in enumerate(POSE_IDX)}).to_numpy()
    pose[fi, li] = d[["x", "y", "z"]].to_numpy(np.float32)
    return resample(hands, T), resample(pose, T), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_signs")
    ap.add_argument("--signs", type=int, default=20, help="keep the N most common signs")
    ap.add_argument("--frames", type=int, default=32, help="resample every clip to this many frames")
    ap.add_argument("--max-per-sign", type=int, default=0, help="cap clips per sign (0 = all)")
    ap.add_argument("--out", default="cache/asl_signs_20.npz")
    args = ap.parse_args()

    meta = pd.read_csv(os.path.join(args.data, "train.csv"))
    counts = meta["sign"].value_counts()
    keep = list(counts.index[:args.signs])
    sub = meta[meta["sign"].isin(keep)]
    if args.max_per_sign:
        sub = sub.groupby("sign", group_keys=False).head(args.max_per_sign)
    print(f"{len(sub)} clips over {len(keep)} signs: {keep}")
    print(f"participants: {sorted(sub['participant_id'].unique())}")

    H, P, N, Y, PID, SID = [], [], [], [], [], []
    t0 = time.time()
    for k, row in enumerate(sub.itertuples(index=False), 1):
        try:
            h, p, n = load_clip(os.path.join(args.data, row.path), args.frames)
        except Exception as e:  # a corrupt or missing file should not kill a 20-minute run
            print(f"  skip {row.path}: {e}")
            continue
        H.append(h.astype(np.float16)); P.append(p.astype(np.float16)); N.append(n)
        Y.append(row.sign); PID.append(int(row.participant_id)); SID.append(int(row.sequence_id))
        if k % 500 == 0 or k == len(sub):
            print(f"  {k}/{len(sub)}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, hands=np.stack(H), pose=np.stack(P), n_frames=np.array(N),
                        sign=np.array(Y), participant=np.array(PID), sequence_id=np.array(SID),
                        signs=np.array(keep))
    mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved {args.out}: {len(Y)} clips, {mb:.1f} MB, median clip length {int(np.median(N))} frames")
    visible = (~np.isnan(np.stack(H).astype(np.float32)[..., 0, 0])).astype(float)  # (N, T, 2): wrist present
    print(f"left hand visible in {visible[:, :, 0].mean() * 100:.0f}% of frames, right hand {visible[:, :, 1].mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
