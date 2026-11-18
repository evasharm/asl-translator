"""
One-time fix for recordings made before record_signs.py corrected for the
webcam's aspect ratio. Rescales landmark x and z into frame-height units
(features.to_square_coords) and stores frame_w / frame_h in the file.

  python fix_aspect.py data_own/eva_room.npz

The frame size is read from the snapshots in data_own/<signer>_<session>/, or
pass it yourself with --size 1920x1080. The original is kept in
data_own/backup/. Files that are already fixed are left alone.
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from features import to_square_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--size", default=None, help="frame size as WIDTHxHEIGHT, e.g. 1920x1080")
    args = ap.parse_args()
    for path in args.npz:
        d = dict(np.load(path, allow_pickle=False))
        if "frame_w" in d:
            print(f"{path}: already fixed, skipping")
            continue
        if args.size:
            w, h = (int(v) for v in args.size.lower().split("x"))
        else:
            snaps = sorted(glob.glob(os.path.join(path[:-4], "*.jpg")))
            if not snaps:
                raise SystemExit(f"{path}: no snapshots in {path[:-4]}/ to read the frame size from, pass --size WxH")
            h, w = cv2.imread(snaps[0]).shape[:2]
        backup = os.path.join(os.path.dirname(path), "backup")
        os.makedirs(backup, exist_ok=True)
        shutil.copy2(path, os.path.join(backup, os.path.basename(path)))
        n = len(d["label"])
        d["landmarks"] = to_square_coords(d["landmarks"], w, h)
        d["frame_w"], d["frame_h"] = np.full(n, w), np.full(n, h)
        np.savez_compressed(path, **d)
        print(f"{path}: fixed {n} frames for a {w}x{h} frame (original in {backup}/)")


if __name__ == "__main__":
    main()
