"""
Landmark preprocessing for the ASL baseline.

Input per sample: 21 MediaPipe hand landmarks x (x, y, z) = 63 values in
MediaPipe's normalized image coordinates, plus a handedness label.

Two feature views are produced:

  raw         the 63 coordinates exactly as MediaPipe returns them (baseline)
  normalized  wrist-centered, scaled by hand size, mirrored to a canonical hand
              (iteration 1)

The normalized view makes the classifier invariant to where the hand sits in
the frame, how far it is from the camera, and which hand is used.
"""

import numpy as np

WRIST = 0
MIDDLE_MCP = 9  # base knuckle of the middle finger


def normalize_landmarks(lm, handedness=None, mirror_left=True):
    """lm: (N, 21, 3) array. handedness: optional (N,) array of 'Left'/'Right'.

    Returns (N, 21, 3) float32 array where, for each sample:
      * the wrist is at the origin,
      * the wrist -> middle-knuckle distance is 1,
      * if mirror_left, left hands are mirrored in x so every sample looks like
        a right hand.
    """
    lm = np.asarray(lm, dtype=np.float32).reshape(-1, 21, 3).copy()
    lm -= lm[:, WRIST:WRIST + 1, :]
    scale = np.linalg.norm(lm[:, MIDDLE_MCP, :2], axis=-1, keepdims=True)  # (N, 1)
    scale = np.where(scale < 1e-6, 1.0, scale)
    lm /= scale[:, :, None]
    if mirror_left and handedness is not None:
        is_left = np.asarray(handedness) == "Left"
        lm[is_left, :, 0] *= -1.0
    return lm


def rotation_normalize(lm):
    """Rotate each hand in the image plane so the wrist -> middle-knuckle
    vector points straight up. Removes in-plane hand rotation. lm must already
    be wrist-centered. (N, 21, 3) -> (N, 21, 3)."""
    lm = np.asarray(lm, dtype=np.float32).reshape(-1, 21, 3).copy()
    v = lm[:, MIDDLE_MCP, :2]
    ang = np.arctan2(v[:, 0], -v[:, 1])  # angle away from "up" (-y in image coords)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x, y = lm[..., 0].copy(), lm[..., 1].copy()
    lm[..., 0] = c * x - s * y
    lm[..., 1] = s * x + c * y
    return lm


def pairwise_distances(lm):
    """Optional extra features: distances between every pair of landmarks.
    lm: (N, 21, 3) -> (N, 210). Scale-free once lm is normalized."""
    lm = np.asarray(lm, dtype=np.float32).reshape(-1, 21, 3)
    iu = np.triu_indices(21, k=1)
    diff = lm[:, :, None, :] - lm[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return d[:, iu[0], iu[1]]


def build_features(lm, handedness=None, view="raw"):
    """Return a (N, D) feature matrix for the requested view."""
    lm = np.asarray(lm, dtype=np.float32).reshape(-1, 21, 3)
    if view == "raw":
        return lm.reshape(len(lm), -1)
    if view == "normalized":
        return normalize_landmarks(lm, handedness).reshape(len(lm), -1)
    if view == "normalized+dist":
        n = normalize_landmarks(lm, handedness)
        return np.concatenate([n.reshape(len(n), -1), pairwise_distances(n)], axis=1)
    if view == "normalized+rot":
        return rotation_normalize(normalize_landmarks(lm, handedness)).reshape(len(lm), -1)
    raise ValueError(f"unknown feature view: {view}")
