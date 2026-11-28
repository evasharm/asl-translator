"""
Iteration 8: turn frame-by-frame letter predictions into a caption, and score it.

Input: data_words/<signer>.npz from record_words.py (each of us fingerspelling
HELLO, BOOK, PENN, CAT, DOG, NAME, WATER, FIVE, TAXI, GOOD).

Letter model: letters.py (midterm recipe). For each person it is trained on the
Kaggle photos + the OTHER person's alphabet recordings only, so nobody is ever
captioned by a model that has seen them.

Caption rules, each one adds to the one before:

  1 raw        argmax of every frame with a hand, consecutive repeats collapsed
  2 vote       majority vote over the last W frames (the midterm demo)
  3 weighted   sum of class probabilities over the last W frames, frames whose
               top probability is below c_min are ignored
  4 +hold      a letter is written only once the weighted label has been the
               same for H frames in a row. The same letter can be written again
               (double letters: HELLO, BOOK, GOOD) only if the wrist moved at
               least d_min hand-lengths between the two holds
  5 +space     G or more frames with no hand -> word break (a space)

Scores: character error rate (CER) = edit distance / reference length.
  per word  each clip on its own, spaces ignored, mean over the 10 words
  stream    all 10 clips of a person joined in order, reference
            "HELLO BOOK ... GOOD", so missing or extra spaces count

Rule parameters are tuned on one person and tested on the other (both ways),
so no number is fitted to the clips it is scored on.

Also runs the whole thing with single-pass landmarks (landmarks_full) instead of
two-stage (landmarks) for iteration 9.

  python eval_captions.py
Outputs: results_caption/results.md, results.json, captions.json (input for eval_cleanup.py)
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from letters import load_letters, predict_proba, train_letters

RULES = ["raw", "vote", "weighted", "hold", "space"]
RULE_NAMES = {"raw": "1 raw argmax", "vote": "2 majority vote", "weighted": "3 confidence-weighted vote",
              "hold": "4 + hold rule", "space": "5 + space on hand drop"}
GRID = {  # parameters each rule adds; a rule is tuned over its own and all earlier ones
    "raw": {},
    "vote": {"W": [4, 8, 12, 16]},
    "weighted": {"W": [4, 8, 12, 16], "c_min": [0.0, 0.3, 0.5]},
    "hold": {"W": [4, 8, 12], "c_min": [0.0, 0.3, 0.5], "H": [6, 10, 14, 18, 22, 26, 30], "d_min": [0.3, 0.6, 1.0, 9.0]},
    "space": {"G": [2, 4, 8]},
}


# ----------------------------------------------------------------------------- scoring

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    return edit_distance(hyp, ref) / max(1, len(ref))


# ----------------------------------------------------------------------------- decoding

def wrist_and_size(lm):
    """lm (N, 21, 3) in frame-height units -> wrist xy (N, 2), hand length (N,)."""
    w = lm[:, 0, :2]
    size = np.linalg.norm(lm[:, 9, :2] - w, axis=1)
    return w, np.where(size < 1e-6, 1.0, size)


def decode(P, det, lm, classes, rule, p):
    """P: (N, C) letter probabilities (rows without a hand are ignored), det: (N,) bool,
    lm: (N, 21, 3). Returns the caption as an upper-case string."""
    level = RULES.index(rule)
    wrist, size = wrist_and_size(lm)
    out, hist = [], []
    last_emit, last_emit_pos = None, None
    cand, cand_n, cand_pos = None, 0, []
    gap = 0
    for t in range(len(det)):
        if not det[t]:
            gap += 1
            if level >= 4 and gap == p["G"] and out and out[-1] != " ":
                out.append(" ")
                last_emit, cand, cand_n = None, None, 0
            continue
        gap = 0
        if level == 0:
            lab = int(P[t].argmax())
        elif level == 1:
            hist = (hist + [int(P[t].argmax())])[-p["W"]:]
            lab = max(set(hist), key=hist.count)
        else:
            if P[t].max() >= p["c_min"]:
                hist = (hist + [P[t]])[-p["W"]:]
            if not hist:
                continue
            lab = int(np.sum(hist, axis=0).argmax())
        if level <= 2:  # emit on every change
            if lab != last_emit:
                out.append(classes[lab])
                last_emit = lab
            continue
        # hold rule: the label must stay the same for H frames
        if lab == cand:
            cand_n += 1
            cand_pos.append(wrist[t] / size[t])
        else:
            cand, cand_n, cand_pos = lab, 1, [wrist[t] / size[t]]
        if cand_n == p["H"]:
            pos = np.mean(cand_pos, axis=0)
            moved = last_emit_pos is not None and np.linalg.norm(pos - last_emit_pos) >= p["d_min"]
            if cand != last_emit or moved:
                out.append(classes[cand])
                last_emit = cand
            last_emit_pos = pos
    return "".join(out).strip().upper()


def all_params(rule):
    keys, vals = [], []
    for r in RULES[:RULES.index(rule) + 1]:
        for k, v in GRID[r].items():
            if k in keys:
                vals[keys.index(k)] = v
            else:
                keys.append(k); vals.append(v)
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)] or [{}]


def caption_person(S, rule, p):
    words, per_word = [], {}
    for w, (P, det, lm) in S["clips"].items():
        per_word[w] = decode(P, det, lm, S["classes"], rule, p)
        words.append(w)
    stream = decode(S["stream"][0], S["stream"][1], S["stream"][2], S["classes"], rule, p)
    return per_word, stream


def score(S, rule, p):
    per_word, stream = caption_person(S, rule, p)
    word_cer = float(np.mean([cer(per_word[w].replace(" ", ""), w) for w in per_word]))
    stream_cer = cer(" ".join(stream.split()), " ".join(per_word))
    return word_cer, stream_cer, per_word, stream


# ----------------------------------------------------------------------------- main

def prepare(signer, key, seed):
    """Train the letter model without `signer`, predict every frame of their word clips."""
    D = load_letters(key=key)
    model, n_train = train_letters(D, D["signer"] != signer, seed=seed)
    d = np.load(f"data_words/{signer}.npz", allow_pickle=False)
    order = np.lexsort((d["frame_index"], d["clip"]))
    lm, det, hands, word = d[key][order], d["detected"][order], d["handedness"][order], d["word"][order]
    P = np.zeros((len(det), len(model.classes_)), np.float32)
    P[det] = predict_proba(model, lm[det], hands[det])
    classes = np.array([c.upper() for c in model.classes_])
    clips = {}
    for w in dict.fromkeys(word.tolist()):
        m = word == w
        clips[w] = (P[m], det[m], lm[m])
    return dict(signer=signer, classes=classes, clips=clips, stream=(P, det, lm), n_train=n_train,
                frames=int(len(det)), det_rate=float(det.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signers", default="eva,isha")
    ap.add_argument("--keys", default="landmarks,landmarks_full",
                    help="landmarks = two-stage (main result), landmarks_full = single pass (iteration 9)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_caption")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    signers = args.signers.split(",")
    assert len(signers) == 2, "tune-on-one / test-on-the-other needs exactly two signers"

    results, captions = {}, {}
    for key in args.keys.split(","):
        t0 = time.time()
        S = {s: prepare(s, key, args.seed) for s in signers}
        print(f"[{key}] letter models trained and frames predicted ({time.time() - t0:.0f}s)")
        res = {}
        for rule in RULES:
            res[rule] = {}
            for test in signers:
                tune = [s for s in signers if s != test][0]
                cands = all_params(rule)
                # tune on the other person: stream CER, then per-word CER as tie-break
                best_p = min(cands, key=lambda p: score(S[tune], rule, p)[1::-1])
                wc, sc, per_word, stream = score(S[test], rule, best_p)
                res[rule][test] = dict(params=best_p, word_cer=wc, stream_cer=sc, per_word=per_word, stream=stream)
                print(f"  {RULE_NAMES[rule]:28s} test {test:5s} word CER {wc * 100:5.1f}%  stream CER {sc * 100:5.1f}%  "
                      f"params {best_p}")
        results[key] = dict(rules=res, info={s: dict(n_train=S[s]["n_train"], frames=S[s]["frames"],
                                                     det_rate=S[s]["det_rate"]) for s in signers})
        if key == "landmarks":
            final = res[RULES[-1]]
            captions = {s: dict(words={w: dict(reference=w, raw=final[s]["per_word"][w]) for w in final[s]["per_word"]},
                                stream=dict(reference=" ".join(final[s]["per_word"]), raw=final[s]["stream"]),
                                rule=RULE_NAMES[RULES[-1]], params=final[s]["params"]) for s in signers}

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    if captions:
        with open(os.path.join(args.out, "captions.json"), "w") as f:
            json.dump(captions, f, indent=2)

    # ------------------------------------------------------------------ report
    L = []
    main_key = "landmarks" if "landmarks" in results else list(results)[0]
    R = results[main_key]["rules"]
    L += ["## Iteration 8: caption rules (two-stage landmarks)", "",
          "Character error rate, lower is better. Parameters tuned on the other person.", "",
          "| Rule | " + " | ".join(f"{s} word CER | {s} stream CER" for s in signers) + " | Mean word CER |",
          "|---|" + "---|---|" * len(signers) + "---|"]
    for rule in RULES:
        cells = " | ".join(f"{R[rule][s]['word_cer'] * 100:.1f}% | {R[rule][s]['stream_cer'] * 100:.1f}%" for s in signers)
        mean = np.mean([R[rule][s]["word_cer"] for s in signers])
        L.append(f"| {RULE_NAMES[rule]} | {cells} | {mean * 100:.1f}% |")
    L += ["", "Parameters chosen (tuned on the other person):", ""]
    for s in signers:
        L.append(f"- test {s}: {R[RULES[-1]][s]['params']}")
    L += ["", "Captions per word (rule 1 raw -> rule 5 final):", "",
          "| Word | " + " | ".join(f"{s} raw | {s} final" for s in signers) + " |", "|---|" + "---|---|" * len(signers)]
    for w in R["raw"][signers[0]]["per_word"]:
        cells = []
        for s in signers:
            raw = R["raw"][s]["per_word"].get(w, "")
            raw = raw if len(raw) <= 24 else raw[:21] + "..."
            cells += [f"`{raw}`", f"`{R[RULES[-1]][s]['per_word'].get(w, '')}`"]
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L += ["", "Streams (all 10 clips joined, final rule):", ""]
    for s in signers:
        L.append(f"- {s}: `{R[RULES[-1]][s]['stream']}`")
    if len(results) > 1:
        L += ["", "## Iteration 9b: single pass vs two-stage on the word recordings", "",
              "| Landmarks | " + " | ".join(f"{s} word CER (final rule)" for s in signers) + " | Mean |", "|---|" + "---|" * (len(signers) + 1)]
        for key, name in [("landmarks_full", "single pass"), ("landmarks", "two-stage")]:
            if key in results:
                RR = results[key]["rules"][RULES[-1]]
                L.append(f"| {name} | " + " | ".join(f"{RR[s]['word_cer'] * 100:.1f}%" for s in signers) +
                         f" | {np.mean([RR[s]['word_cer'] for s in signers]) * 100:.1f}% |")
    info = results[main_key]["info"]
    L += ["", "Frames per person: " + ", ".join(f"{s} {info[s]['frames']} ({info[s]['det_rate'] * 100:.0f}% with a hand)" for s in signers)]
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
