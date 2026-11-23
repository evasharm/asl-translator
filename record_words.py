"""
Record fingerspelled words for the captioning experiments (iterations 8 and 10).

The window shows one word at a time, counts down, then records landmarks while
you fingerspell it letter by letter. Drop your hand out of the frame when the
word is done (or press n). Each word is one clip.

  python record_words.py --signer isha
  python record_words.py --signer eva --words HELLO,BOOK
  python record_words.py --signer isha --start-from NAME      # resume

Keys:  n = word done, go to the next one   r = redo this word   q = quit (saves)

Output: data_words/<signer>.npz with one row per frame:
  landmarks (N, 21, 3)  two-stage landmarks in frame-height units
  landmarks_full        single pass landmarks
  proba? no: raw landmarks only, the classifier runs later in eval_captions.py
  detected, handedness, word, clip (0..W-1), frame_index, frame_w, frame_h
Also data_words/<signer>/<WORD>.jpg, one snapshot per word.
"""

import argparse
import os
import time

import cv2
import numpy as np

import hand_landmarks_demo as hd

hd._ensure = hd.ensure_model
from features import to_square_coords
from twostage import TwoStage, full_frame_points

DEFAULT_WORDS = ["HELLO", "BOOK", "PENN", "CAT", "DOG", "NAME", "WATER", "FIVE", "TAXI", "GOOD"]


def put(frame, text, y, scale=1.0, color=(255, 255, 255)):
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True)
    ap.add_argument("--words", default=",".join(DEFAULT_WORDS))
    ap.add_argument("--max-seconds", type=float, default=25.0, help="safety limit per word")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data_words")
    ap.add_argument("--min-size", type=float, default=0.35)
    args = ap.parse_args()

    model_path = os.environ.get("HAND_MODEL", "models/hand_landmarker.task")
    if not os.path.exists(model_path):
        model_path = "hand_landmarker.task"
    hd.MODEL_PATH = model_path
    hd.ensure_model = lambda path=model_path: hd.__dict__["_ensure"](path)
    hd.SWAP_LABELS = False

    words = [w.strip().upper() for w in args.words.split(",") if w.strip()]
    if args.start_from:
        words = words[words.index(args.start_from.upper()):]
    os.makedirs(args.out, exist_ok=True)
    snap_dir = os.path.join(args.out, args.signer)
    os.makedirs(snap_dir, exist_ok=True)
    npz_path = os.path.join(args.out, f"{args.signer}.npz")

    rows = []
    if os.path.exists(npz_path):
        old = np.load(npz_path, allow_pickle=False)
        for i in range(len(old["word"])):
            if old["word"][i] not in words:
                rows.append((old["landmarks"][i], old["landmarks_full"][i], bool(old["detected"][i]), str(old["handedness"][i]),
                             str(old["word"][i]), int(old["clip"][i]), int(old["frame_index"][i]), int(old["frame_w"][i]), int(old["frame_h"][i])))
        print(f"Resuming {npz_path}: keeping {len(rows)} frames for other words")

    def save():
        if not rows:
            return
        lm, lmf, det, hands, wd, clip, fi, fw, fh = zip(*rows)
        np.savez_compressed(npz_path, landmarks=np.stack(lm).astype(np.float32), landmarks_full=np.stack(lmf).astype(np.float32),
                            detected=np.array(det), handedness=np.array(hands), word=np.array(wd), clip=np.array(clip),
                            frame_index=np.array(fi), frame_w=np.array(fw), frame_h=np.array(fh),
                            signer=np.array([args.signer] * len(wd)))
        print(f"Saved {npz_path}: {len(wd)} frames, {len(set(wd))} words")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")
    last_ts, t_start = -1, time.time()
    with TwoStage(model_path) as ts2:
        wi = 0
        while wi < len(words):
            word = words[wi]
            phase, phase_t0, frame_i, got = "countdown", time.time(), 0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                ts = max(int((time.time() - t_start) * 1000), last_ts + 1)
                last_ts = ts
                result = ts2.detect_video(frame, ts)
                elapsed = time.time() - phase_t0
                H, W = frame.shape[:2]
                size = 0.0
                if result.hand_landmarks:
                    ys_ = [p.y for p in result.hand_landmarks[0]]
                    size = max(ys_) - min(ys_)
                    if size < args.min_size:
                        put(frame, "MOVE CLOSER: hand is too small in the frame", 140, 1.0, (0, 0, 255))
                if phase == "countdown":
                    put(frame, f"Next word: {word}   ({wi + 1}/{len(words)})", 50, 1.3, (80, 220, 255))
                    put(frame, f"get ready... {max(0, args.countdown - elapsed):.0f}   spell it letter by letter, then drop your hand", 95, 0.8)
                    if elapsed >= args.countdown:
                        phase, phase_t0 = "record", time.time()
                else:
                    detected = bool(result.hand_landmarks)
                    if detected:
                        pts_full, _ = full_frame_points(result)
                        pts, _, box = ts2.refine(frame, result)
                        pts, pts_full = to_square_coords(pts, W, H), to_square_coords(pts_full, W, H)
                        hand = hd.hand_label(result.handedness[0])
                        if box is not None:
                            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[2]), (80, 255, 120), 2)
                        if got == 0:
                            cv2.imwrite(os.path.join(snap_dir, f"{word}.jpg"), frame)
                        got += 1
                    else:
                        pts = pts_full = np.zeros((21, 3), np.float32)
                        hand = ""
                    rows.append((pts, pts_full, detected, hand, word, wi, frame_i, W, H))
                    frame_i += 1
                    put(frame, f"Spell: {word}   then drop your hand and press n", 50, 1.3, (80, 255, 120))
                    put(frame, f"recording {elapsed:.0f}s   frames with hand: {got}   hand size {size:.2f}", 95, 1.0)
                    if elapsed >= args.max_seconds:
                        break
                hd.draw_hands(frame, result)
                put(frame, "n = next word   r = redo   q = quit", H - 16, 0.7, (200, 200, 200))
                cv2.imshow("Record words", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    save(); cap.release(); cv2.destroyAllWindows(); return
                if key == ord("n") and phase == "record":
                    break
                if key == ord("r"):
                    rows[:] = [r for r in rows if r[4] != word]
                    phase, phase_t0, frame_i, got = "countdown", time.time(), 0, 0
            wi += 1
            save()
    cap.release()
    cv2.destroyAllWindows()
    save()
    print("Done.")


if __name__ == "__main__":
    main()
