"""
Two-stage hand landmarks: detect on the full frame, then crop around the hand
and run the landmarker again on the crop.

Why: the Kaggle training images are hands that fill the whole picture, but on a
webcam the hand is often a small fraction of the frame, and MediaPipe's
landmarks (especially the thumb and the z depth) get noisy on a small hand. On
our first recordings accuracy went from 17% for the smallest hands to 43% for
the largest. Re-running the landmarker on a tight crop gives it a big hand
again, like the training data.

Usage inside a frame loop (see record_signs.py and demo_predict.py):

    ts = TwoStage(model_path)
    result = ts.detect_video(frame_bgr, timestamp_ms)   # full-frame, as before
    pts, hand, box = ts.refine(frame_bgr, result)       # (21, 3) in full-frame
                                                        # normalized coords, or None
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class TwoStage:
    def __init__(self, model_path, min_conf=0.5, margin=0.5, crop_size=512):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        self.video = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf))
        self.image = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf))
        self.margin = margin
        self.crop_size = crop_size

    def close(self):
        self.video.close()
        self.image.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def detect_video(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.video.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(timestamp_ms))

    def detect_image(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.image.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    def crop_box(self, result, W, H):
        """Square crop box (x0, y0, side) in pixels around the first hand, with margin."""
        lm = result.hand_landmarks[0]
        xs = np.array([p.x for p in lm]) * W
        ys = np.array([p.y for p in lm]) * H
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * self.margin)
        side = max(int(side), 64)
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        return x0, y0, side

    def refine(self, frame_bgr, result):
        """Second pass on a crop. Returns (pts, handedness, box) with pts (21, 3) in
        full-frame normalized coordinates (x by width, y by height, z like x), or
        (None, "", None) if no hand. Falls back to the full-frame landmarks if the
        crop pass finds nothing."""
        if not result.hand_landmarks:
            return None, "", None
        H, W = frame_bgr.shape[:2]
        hand = result.handedness[0][0].category_name
        x0, y0, side = self.crop_box(result, W, H)
        # pad the frame so the crop can hang off the edge
        pad = side
        padded = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        if crop.size == 0:
            pts = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], np.float32)
            return pts, hand, (x0, y0, side)
        if side != self.crop_size:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA if side > self.crop_size else cv2.INTER_LINEAR)
        r2 = self.detect_image(crop)
        if not r2.hand_landmarks:
            pts = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], np.float32)
            return pts, hand, (x0, y0, side)
        c = np.array([(p.x, p.y, p.z) for p in r2.hand_landmarks[0]], np.float32)
        pts = np.empty_like(c)
        pts[:, 0] = (x0 + c[:, 0] * side) / W
        pts[:, 1] = (y0 + c[:, 1] * side) / H
        pts[:, 2] = c[:, 2] * side / W  # z is in crop-width units, convert to frame-width units
        return pts, hand, (x0, y0, side)


def full_frame_points(result):
    if not result.hand_landmarks:
        return None, ""
    return (np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], np.float32),
            result.handedness[0][0].category_name)
