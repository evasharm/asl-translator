"""
Hand landmark extraction demo for the ASL captioning project.

Runs MediaPipe's HandLandmarker on a webcam, a video file, or a single image,
draws the 21 landmarks per hand on each frame, and saves:

  demo_out/annotated.mp4   annotated video (webcam and video sources)
  demo_out/snapshot.png    one annotated frame, ready to drop into the slide
  demo_out/landmarks.npy   array of shape (num_frames, 126)
                           [left hand 21 x (x, y, z), right hand 21 x (x, y, z)]
                           a hand that is not detected is filled with zeros

Setup:
  pip install mediapipe opencv-python numpy

Usage:
  python hand_landmarks_demo.py                      # webcam 0, live window
  python hand_landmarks_demo.py --source clip.mp4    # video file
  python hand_landmarks_demo.py --source frame.jpg   # single image
  python hand_landmarks_demo.py --source clip.mp4 --no-display

Keys in the live window:  q = quit,  s = save the current frame as the snapshot
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Standard MediaPipe hand topology. Index 0 is the wrist, then 4 joints per finger.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]
# BGR colors per hand label
HAND_COLORS = {"Left": (75, 138, 240), "Right": (255, 176, 127)}

# MediaPipe's Left/Right labels assume a mirrored (selfie) image. For a normal,
# unmirrored video or photo the labels come out swapped, so main() turns this on.
SWAP_LABELS = False


def hand_label(handed):
    name = handed[0].category_name
    if SWAP_LABELS:
        name = "Right" if name == "Left" else "Left"
    return name


def ensure_model(path=MODEL_PATH):
    """Download the HandLandmarker model once (about 8 MB)."""
    if not os.path.exists(path):
        print(f"Downloading hand landmarker model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(mode, max_hands, min_conf):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=mode,
        num_hands=max_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_feature_vector(result):
    """Pack one frame's detections into a fixed 126-dim vector.

    Slot 0 (first 63 values) holds the hand labeled Left, slot 1 the hand
    labeled Right. If both hands get the same label, the second one takes the
    free slot. Coordinates are MediaPipe's normalized image coordinates.
    """
    vec = np.zeros((2, 21, 3), dtype=np.float32)
    used = [False, False]
    for landmarks, handed in zip(result.hand_landmarks, result.handedness):
        slot = 0 if hand_label(handed) == "Left" else 1
        if used[slot]:
            slot = 1 - slot
        if used[slot]:
            continue
        vec[slot] = [(lm.x, lm.y, lm.z) for lm in landmarks]
        used[slot] = True
    return vec.reshape(-1)


def draw_hands(frame, result):
    """Draw the skeleton, the 21 points, and the handedness label on a BGR frame."""
    h, w = frame.shape[:2]
    thickness = max(2, w // 400)
    radius = max(3, w // 250)
    for landmarks, handed in zip(result.hand_landmarks, result.handedness):
        label = hand_label(handed)
        color = HAND_COLORS.get(label, (255, 255, 255))
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (245, 245, 245), thickness, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, p, radius, (40, 40, 40), 1, cv2.LINE_AA)
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        text, org = f"{label} {handed[0].score:.2f}", (x0, max(70, y0 - 14))
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 6, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def hand_area(result):
    """Largest hand bounding box area in normalized units, used to pick a snapshot."""
    best = 0.0
    for landmarks in result.hand_landmarks:
        xs, ys = [lm.x for lm in landmarks], [lm.y for lm in landmarks]
        best = max(best, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    return best


def draw_hud(frame, n_hands, fps):
    text = f"hands: {n_hands}   landmarks: {n_hands * 21}   {fps:4.1f} FPS"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (58, 35, 19), -1)
    cv2.putText(frame, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (245, 245, 245), 2, cv2.LINE_AA)
    return frame


def detect(landmarker, frame_bgr, timestamp_ms=None):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        return landmarker.detect(image)
    return landmarker.detect_for_video(image, int(timestamp_ms))


def run_image(args):
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.source}")
    with make_landmarker(vision.RunningMode.IMAGE, args.max_hands, args.min_conf) as lm:
        result = detect(lm, frame)
    vec = to_feature_vector(result)
    draw_hands(frame, result)
    cv2.imwrite(os.path.join(args.out, "snapshot.png"), frame)
    np.save(os.path.join(args.out, "landmarks.npy"), vec[None, :])
    print(f"Hands detected: {len(result.hand_landmarks)}")
    print(f"Feature vector shape: {vec.shape}")
    if not args.no_display:
        cv2.imshow("Hand landmarks", frame)
        cv2.waitKey(0)


def run_stream(args):
    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0

    writer = None
    vectors = []
    frames_with_hand = 0
    snapshot_locked = False  # True once the user picks a frame with the s key
    best_area = 0.0
    snapshot_path = os.path.join(args.out, "snapshot.png")
    frame_idx = 0
    t_start = time.time()
    fps = 0.0
    last_ts = -1

    with make_landmarker(vision.RunningMode.VIDEO, args.max_hands, args.min_conf) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                # Mirror so the window behaves like a mirror.
                frame = cv2.flip(frame, 1)
                ts = (time.time() - t_start) * 1000.0
            else:
                ts = frame_idx * 1000.0 / src_fps
            ts = max(int(ts), last_ts + 1)  # timestamps must strictly increase
            last_ts = ts

            t0 = time.time()
            result = detect(lm, frame, ts)
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps and dt > 0 else (1.0 / dt if dt > 0 else 0.0)

            n_hands = len(result.hand_landmarks)
            vectors.append(to_feature_vector(result))
            frames_with_hand += int(n_hands > 0)

            draw_hands(frame, result)
            draw_hud(frame, n_hands, fps)

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(frame)

            # Default snapshot: the frame where the hand appears largest.
            area = hand_area(result)
            if not snapshot_locked and area > best_area:
                best_area = area
                cv2.imwrite(snapshot_path, frame)

            if not args.no_display:
                cv2.imshow("Hand landmarks (q = quit, s = snapshot)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    cv2.imwrite(snapshot_path, frame)
                    snapshot_locked = True
                    print(f"Saved snapshot to {snapshot_path}")
            frame_idx += 1

    elapsed = time.time() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if not vectors:
        raise SystemExit("No frames were read from the source.")
    data = np.stack(vectors)
    np.save(os.path.join(args.out, "landmarks.npy"), data)

    n = len(vectors)
    print("\n=== Summary (numbers for the slide) ===")
    print(f"Frames processed:        {n}")
    print(f"Frames with a hand:      {frames_with_hand} ({100.0 * frames_with_hand / n:.1f}%)")
    print(f"Model speed (smoothed):  {fps:.1f} FPS")
    print(f"End to end throughput:   {n / elapsed:.1f} FPS")
    print(f"Landmark array shape:    {data.shape}  -> {os.path.join(args.out, 'landmarks.npy')}")
    print(f"Outputs saved in:        {args.out}/")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe hand landmark extraction demo")
    parser.add_argument("--source", default="0",
                        help="webcam index (default 0), or path to a video or image")
    parser.add_argument("--out", default="demo_out", help="output folder")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--mirrored", action="store_true",
                        help="the video or image file is already mirrored (selfie style)")
    parser.add_argument("--no-display", action="store_true",
                        help="do not open a window, only write the output files")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    global SWAP_LABELS
    SWAP_LABELS = not args.source.isdigit() and not args.mirrored

    if args.source.lower().endswith(IMAGE_EXTS):
        run_image(args)
    else:
        run_stream(args)


if __name__ == "__main__":
    main()
