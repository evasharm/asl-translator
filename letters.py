"""
Shared letter-classifier helpers for iterations 8 and 9 (and the final model).

The letter model is the midterm recipe (train_own.py): normalized xy landmarks,
tilt + jitter augmentation, logistic regression. Two things differ because the
output is now a caption, not a 36-way class:

  * letters only. Digits that share a handshape with a letter are merged into
    that letter (0->O, 2->V, 6->W, 9->F); the other digits are dropped. J and Z
    are dropped as before (they are motion letters).
  * every recording is capped per letter so 15k near-duplicate webcam frames
    do not swamp 1.5k Kaggle photos (same idea as --max-own-per-sign 40).
"""

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from augment import augment
from features import build_features

VIEW = "normalized_xy"
MERGE = {"0": "o", "2": "v", "6": "w", "9": "f"}
DROP = {"j", "z", "1", "3", "4", "5", "7", "8"}


def to_letter(labels):
    labels = np.array([MERGE.get(str(t).lower(), str(t).lower()) for t in labels])
    return labels


def load_letters(cache="cache/train_landmarks.npz", own_glob="data_own/*.npz", key="landmarks"):
    """Kaggle photos + every own alphabet recording. key = 'landmarks' (two-stage in
    v2 recordings, single pass in v1) or 'landmarks_full' (single pass, v2 only).
    Returns a dict of flat arrays; 'rec' names the file each row came from."""
    parts = []
    d = np.load(cache, allow_pickle=False)
    parts.append(dict(lm=d["landmarks"], det=d["detected"], hands=d["handedness"], y=to_letter(d["label"]),
                      signer=np.char.lower(d["signer"].astype(str)), rec=np.array(["kaggle"] * len(d["label"]))))
    for p in sorted(glob.glob(own_glob)):
        d = np.load(p, allow_pickle=False)
        lm = d[key] if key in d.files else d["landmarks"]
        name = p.split("/")[-1].replace(".npz", "")
        parts.append(dict(lm=lm, det=d["detected"], hands=d["handedness"], y=to_letter(d["label"]),
                          signer=np.char.lower(d["signer"].astype(str)), rec=np.array([name] * len(d["label"]))))
    D = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    D["keep"] = D["det"] & ~np.isin(D["y"], list(DROP))
    return D


def cap_mask(D, mask, cap):
    """Evenly spaced subset of at most `cap` frames per (recording, letter) inside mask.
    Kaggle photos are never capped."""
    out = mask & (D["rec"] == "kaggle")
    for r in sorted(set(D["rec"][mask]) - {"kaggle"}):
        for c in sorted(set(D["y"])):
            idx = np.where(mask & (D["rec"] == r) & (D["y"] == c))[0]
            if cap and len(idx) > cap:
                idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
            out[idx] = True
    return out


def train_letters(D, mask, seed=0, copies=5, cap=40):
    rng = np.random.default_rng(seed)
    tr = cap_mask(D, mask & D["keep"], cap)
    lm, y, h = augment(D["lm"][tr], D["y"][tr], D["hands"][tr], rng, copies=copies, ops=("tilt", "jitter"))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    model.fit(build_features(lm, h, VIEW), y)
    return model, int(len(y))


def predict_proba(model, lm, hands):
    return model.predict_proba(build_features(lm, hands, VIEW))
