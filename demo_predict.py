"""
Live fingerspelling demo: webcam or video -> MediaPipe landmarks -> trained
classifier -> letter overlay, smoothed with a sliding-window majority vote.

  python demo_predict.py                                   # webcam
  python demo_predict.py --source clip.mp4 --no-display    # video, writes demo_out/predicted.mp4
  python demo_predict.py --model results/model_iter2.joblib
"""

import argparse
import os
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
from mediapipe.tasks.python import vision

import hand_landmarks_demo as hd

hd._ensure = hd.ensure_model  # keep the original before overriding its default path
from features import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="results/model_iter2.joblib")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--window", type=int, default=8, help="frames in the majority-vote window")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model_path = os.environ.get("HAND_MODEL", "models/hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = "hand_landmarker.task"  # hand_landmarks_demo.py downloads it here
    hd.MODEL_PATH = model_path
    hd.ensure_model = lambda path=model_path: hd.__dict__["_ensure"](path)
    bundle = joblib.load(args.model)
    model, view = bundle["model"], bundle["view"]

    is_webcam = args.source.isdigit()
    hd.SWAP_LABELS = not is_webcam
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0

    votes = deque(maxlen=args.window)
    caption, writer = "", None
    t_start, last_ts, frame_idx = time.time(), -1, 0
    with hd.make_landmarker(vision.RunningMode.VIDEO, 1, args.min_conf) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)
                ts = (time.time() - t_start) * 1000.0
            else:
                ts = frame_idx * 1000.0 / src_fps
            ts = max(int(ts), last_ts + 1)
            last_ts = ts
            result = hd.detect(lm, frame, ts)

            label, conf = "-", 0.0
            if result.hand_landmarks:
                pts = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], np.float32)
                hand = hd.hand_label(result.handedness[0])
                X = build_features(pts[None], np.array([hand]), view)
                proba = model.predict_proba(X)[0]
                k = int(proba.argmax())
                label, conf = model.classes_[k], float(proba[k])
                votes.append(label)
            else:
                votes.append("-")
            smoothed, n = Counter(votes).most_common(1)[0]
            if n >= max(2, args.window // 2) and smoothed != "-":
                caption = smoothed

            hd.draw_hands(frame, result)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (58, 35, 19), -1)
            cv2.putText(frame, f"frame: {label} ({conf:.2f})   caption: {caption}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(os.path.join(args.out, "predicted.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(frame)
            if not args.no_display:
                cv2.imshow("ASL fingerspelling (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Wrote {os.path.join(args.out, 'predicted.mp4')}")


if __name__ == "__main__":
    main()
