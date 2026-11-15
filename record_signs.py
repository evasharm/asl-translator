"""
Record your own fingerspelling dataset from the webcam.

The window prompts one sign at a time ("Sign: A"), gives you a 3 second
countdown, then records landmarks for --seconds seconds while you hold the
sign. Slowly rotate and move the hand a little while holding it so the frames
are not all identical. Then it moves to the next sign.

  python record_signs.py --signer isha --session window
  python record_signs.py --signer eva --session room --seconds 12
  python record_signs.py --signer isha --session window --start-from m   # resume

Keys while recording:  n = skip to the next sign now,  r = redo the current sign,
                       q = quit (everything recorded so far is saved)

Output:
  data_own/<signer>_<session>.npz    same format as cache/train_landmarks.npz
                                     (landmarks, detected, handedness, label,
                                      index, path, signer) plus session
  data_own/<signer>_<session>/<label>.jpg   one snapshot per sign for the report

Setup: pip install mediapipe opencv-python numpy, and models/hand_landmarker.task
must exist (hand_landmarks_demo.py downloads it on first run).
"""

import argparse
import os
import time

import cv2
import numpy as np
from mediapipe.tasks.python import vision

import hand_landmarks_demo as hd

hd._ensure = hd.ensure_model  # keep the original before overriding its default path

DEFAULT_CLASSES = list("abcdefghijklmnopqrstuvwxyz") + list("0123456789")


def put(frame, text, y, scale=1.0, color=(255, 255, 255)):
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True, help="your first name, lowercase")
    ap.add_argument("--session", default="s1", help="e.g. window, room, far")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--seconds", type=float, default=15.0, help="recording time per sign")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--start-from", default=None, help="resume from this label")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data_own")
    args = ap.parse_args()

    model_path = os.environ.get("HAND_MODEL", "models/hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = "hand_landmarker.task"  # hand_landmarks_demo.py downloads it here
    hd.MODEL_PATH = model_path
    hd.ensure_model = lambda path=model_path: hd.__dict__["_ensure"](path)
    hd.SWAP_LABELS = False  # webcam is mirrored, MediaPipe labels are right as is

    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    if args.start_from:
        classes = classes[classes.index(args.start_from.lower()):]

    os.makedirs(args.out, exist_ok=True)
    snap_dir = os.path.join(args.out, f"{args.signer}_{args.session}")
    os.makedirs(snap_dir, exist_ok=True)
    npz_path = os.path.join(args.out, f"{args.signer}_{args.session}.npz")

    # Resume: keep anything already recorded in this session file for other labels.
    rows = []
    if os.path.exists(npz_path):
        old = np.load(npz_path, allow_pickle=False)
        for i in range(len(old["label"])):
            if old["label"][i] not in classes:
                rows.append((old["landmarks"][i], bool(old["detected"][i]), str(old["handedness"][i]),
                             str(old["label"][i]), int(old["index"][i])))
        print(f"Resuming {npz_path}: keeping {len(rows)} frames for other signs")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")

    def save():
        if not rows:
            return
        lm, det, hands, labels, idx = zip(*rows)
        np.savez_compressed(npz_path, landmarks=np.stack(lm).astype(np.float32), detected=np.array(det),
                            handedness=np.array(hands), label=np.array(labels), index=np.array(idx),
                            path=np.array([""] * len(labels)), signer=np.array([args.signer] * len(labels)),
                            session=np.array([args.session] * len(labels)))
        n_det = sum(det)
        print(f"Saved {npz_path}: {len(labels)} frames, {n_det} with a hand, "
              f"{len(set(labels))} signs")

    last_ts = -1
    t_start = time.time()
    with hd.make_landmarker(vision.RunningMode.VIDEO, 1, 0.5) as lm:
        ci = 0
        while ci < len(classes):
            label = classes[ci]
            phase, phase_t0, frame_i, got, redo = "countdown", time.time(), 0, 0, False
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                ts = max(int((time.time() - t_start) * 1000), last_ts + 1)
                last_ts = ts
                result = hd.detect(lm, frame, ts)
                hd.draw_hands(frame, result)
                elapsed = time.time() - phase_t0

                if phase == "countdown":
                    put(frame, f"Next sign: {label.upper()}   ({ci + 1}/{len(classes)})", 50, 1.3, (80, 220, 255))
                    put(frame, f"get ready... {max(0, args.countdown - elapsed):.0f}", 95, 1.0)
                    if elapsed >= args.countdown:
                        phase, phase_t0 = "record", time.time()
                else:
                    detected = bool(result.hand_landmarks)
                    if detected:
                        pts = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], np.float32)
                        hand = hd.hand_label(result.handedness[0])
                        if got == 0:
                            cv2.imwrite(os.path.join(snap_dir, f"{label}.jpg"), frame)
                        got += 1
                    else:
                        pts, hand = np.zeros((21, 3), np.float32), ""
                    rows.append((pts, detected, hand, label, frame_i))
                    frame_i += 1
                    put(frame, f"Sign: {label.upper()}   hold and move slowly", 50, 1.3, (80, 255, 120))
                    put(frame, f"recording {max(0, args.seconds - elapsed):.0f}s   frames with hand: {got}", 95, 1.0)
                    if elapsed >= args.seconds:
                        break
                put(frame, "n = next   r = redo   q = quit", frame.shape[0] - 16, 0.7, (200, 200, 200))
                cv2.imshow("Record signs", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    rows[:] = [r for r in rows if not (r[3] == label and phase == "record")] if redo else rows
                    save()
                    cap.release(); cv2.destroyAllWindows()
                    return
                if key == ord("n"):
                    break
                if key == ord("r"):
                    rows[:] = [r for r in rows if r[3] != label]
                    phase, phase_t0, frame_i, got = "countdown", time.time(), 0, 0
            if got < 30 and phase == "record":
                print(f"  warning: only {got} frames with a hand for '{label}', consider redoing with --start-from {label}")
            ci += 1
            save()
    cap.release()
    cv2.destroyAllWindows()
    save()
    print("Done.")


if __name__ == "__main__":
    main()
