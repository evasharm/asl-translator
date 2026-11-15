"""
Landmark augmentation for the ASL fingerspelling classifier.

All functions take and return arrays of shape (N, 21, 3) in MediaPipe's
normalized image coordinates (x, y in [0, 1], z relative depth). Augmentations
are applied around the wrist so they look like a different camera pose or a
slightly different hand, not a shifted image.

  rotate_2d    rotate the hand in the image plane by a random angle
  scale        shrink or grow the hand (distance from the camera)
  jitter       add Gaussian noise to every point (landmark regression noise)
  tilt_3d      rotate about the vertical axis in 3D, then keep x, y, z
  augment      apply a random combination, K copies per sample
"""

import numpy as np

WRIST = 0


def _about_wrist(lm, fn):
    w = lm[:, WRIST:WRIST + 1, :]
    return fn(lm - w) + w


def rotate_2d(lm, rng, max_deg=15.0):
    th = np.deg2rad(rng.uniform(-max_deg, max_deg, size=len(lm)))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]

    def fn(p):
        x, y = p[..., 0], p[..., 1]
        out = p.copy()
        out[..., 0] = c * x - s * y
        out[..., 1] = s * x + c * y
        return out
    return _about_wrist(lm, fn)


def scale(lm, rng, lo=0.8, hi=1.2):
    f = rng.uniform(lo, hi, size=(len(lm), 1, 1))
    return _about_wrist(lm, lambda p: p * f)


def jitter(lm, rng, sigma=0.01):
    # sigma is relative to hand size (wrist -> middle knuckle distance)
    size = np.linalg.norm(lm[:, 9, :2] - lm[:, 0, :2], axis=-1)[:, None, None]
    return lm + rng.normal(0, 1, size=lm.shape) * sigma * size


def tilt_3d(lm, rng, max_deg=25.0):
    """Rotate about the y axis (hand turning left/right toward the camera).
    z from MediaPipe is roughly in the same units as x, so we treat (x, z) as a
    plane and rotate in it."""
    th = np.deg2rad(rng.uniform(-max_deg, max_deg, size=len(lm)))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]

    def fn(p):
        x, z = p[..., 0], p[..., 2]
        out = p.copy()
        out[..., 0] = c * x + s * z
        out[..., 2] = -s * x + c * z
        return out
    return _about_wrist(lm, fn)


def augment(lm, y, handedness, rng, copies=5, ops=("rotate", "scale", "jitter", "tilt")):
    """Return the original samples plus `copies` augmented versions of each.
    Each copy applies every op in `ops` with a fresh random draw."""
    outs, ys, hs = [lm], [y], [handedness]
    for _ in range(copies):
        a = lm.copy()
        if "rotate" in ops:
            a = rotate_2d(a, rng)
        if "scale" in ops:
            a = scale(a, rng)
        if "tilt" in ops:
            a = tilt_3d(a, rng)
        if "jitter" in ops:
            a = jitter(a, rng)
        outs.append(a.astype(np.float32))
        ys.append(y)
        hs.append(handedness)
    return np.concatenate(outs), np.concatenate(ys), np.concatenate(hs)
