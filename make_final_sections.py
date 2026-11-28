"""Draft the new final-report sections (iterations 7 to 10, lessons learned, future work)
as Final_Report_New_Sections.docx, in the same style as the midterm report.

Every number is read from the results files, so rerunning an experiment and then this
script keeps the text in sync:
  results_signs_20/, results_signs_50/   iteration 7 (train_signs.py --grid)
  results_caption/                       iteration 8 (eval_captions.py)
  results_final/                         iteration 9 (train_final.py)
  results_cleanup/*.json                 iteration 10 (eval_cleanup.py; LLM rows appear once Eva runs it)

Text in [square brackets] is for Eva and Isha to fill in or rewrite in their own words.

  python make_final_sections.py . Final_Report_New_Sections.docx
"""
import glob
import json
import os
import sys

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Final_Report_New_Sections.docx")
J = lambda *p: json.load(open(os.path.join(ROOT, *p))) if os.path.exists(os.path.join(ROOT, *p)) else None
pct = lambda x: f"{x * 100:.1f}%"
FONT = "Times New Roman"

S20, S50 = J("results_signs_20", "results.json"), J("results_signs_50", "results.json")
CAP = J("results_caption", "results.json")
FIN = J("results_final", "results.json")
CLEAN = {}
for p in sorted(glob.glob(os.path.join(ROOT, "results_cleanup", "*.json"))):
    d = json.load(open(p))
    CLEAN[d["provider"]] = d

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.7)
    s.left_margin = s.right_margin = Inches(0.85)


def set_font(run, size=12, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold, run.italic = bold, italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(k), FONT)


st = doc.styles["Normal"]
st.font.name, st.font.size = FONT, Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(4)


def H(text):
    p = doc.add_paragraph()
    set_font(p.add_run(text), 12, bold=True)
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(10), Pt(2)
    p.paragraph_format.keep_with_next = True


def H2(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, 12)
    r.underline = True
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(6), Pt(1)
    p.paragraph_format.keep_with_next = True


def P(text, lead=None, size=12):
    p = doc.add_paragraph()
    if lead:
        set_font(p.add_run(lead + " "), size, bold=True)
    set_font(p.add_run(text), size)
    return p


def B(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        set_font(p.add_run(lead + " "), 12, bold=True)
    set_font(p.add_run(text), 12)
    p.paragraph_format.space_after = Pt(2)


def table(rows, widths, size=10):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            set_font(para.add_run(str(val)), size, bold=(r == 0))
            cell.width = Inches(widths[c])
            para.paragraph_format.keep_with_next = (r == 0)
        trpr = t.rows[r]._tr.get_or_add_trPr()
        trpr.append(trpr.makeelement(qn("w:cantSplit"), {}))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(path, width, caption):
    path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        P(f"[FIGURE PLACEHOLDER: {os.path.basename(path)} not generated yet]")
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(c.add_run(caption), 10, italic=True)


def signs_figure(path):
    """Grouped bars: each config, 20 vs 50 signs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["frame-average\nbaseline", "GRU", "GRU\n+velocity", "GRU\n+vel +aug", "GRU\n+vel +aug +arms"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(labels))
    for i, (S, name, color) in enumerate([(S20, "20 signs", "#4c78a8"), (S50, "50 signs", "#f58518")]):
        if not S:
            continue
        acc = [r["accuracy"] * 100 for r in S["rows"]]
        err = [r["accuracy_std"] * 100 for r in S["rows"]]
        ax.bar(x + (i - 0.5) * 0.38, acc, 0.38, yerr=err, capsize=3, label=name, color=color)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("accuracy on unseen signers (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Iteration 7: word signs, held-out participants")
    fig.tight_layout()
    fig.savefig(path, dpi=170)


# ============================================================================ iteration 7
H("Iteration 7: word-level signs")
P("Fingerspelling is slow, so a usable caption system also needs whole-word signs. We used the Google ASL Signs "
  "dataset (Kaggle), which already provides MediaPipe landmarks for about 94,000 clips of 250 signs from 21 signers. "
  "prep_signs.py keeps both hands plus six arm points (shoulders, elbows, wrists) and resamples each clip to 32 frames. "
  "We built two subsets: the 20 most common signs (300 clips each) and 50 signs (200 clips each). Four whole "
  "participants are held out for testing and one more for early stopping, so every number below is on people the model "
  "has never seen, the same rule as hand2 for the letters.")
P("Each frame becomes a feature vector: both hands wrist-centered and scaled by hand length (the letter normalization), "
  "the dominant wrist's position relative to its average in the clip, the other wrist relative to the dominant one, and "
  "a hand-present flag. Clips signed mostly with the left hand are mirrored. The baseline averages these features over "
  "the clip (mean, std, min, max) and uses logistic regression, which throws away the order of the frames. The model is "
  "a two-layer bidirectional GRU (128 units) with mean and max pooling over time. We then add, one at a time: velocity "
  "(frame-to-frame differences), augmentation (random scale, ±12° rotation, and a random 80 to 100% time crop of each clip) and the arm points "
  "from the pose model. 3 seeds per GRU row, 30 epochs, best validation epoch kept.")
if S20 or S50:
    yn = lambda b: "yes" if b else "no"
    rows = [["Model", "Velocity", "Augment", "Arms", "20 signs", "50 signs"]]
    base = (S20 or S50)["rows"]
    for i, r in enumerate(base):
        cells = []
        for S in (S20, S50):
            if S:
                q = S["rows"][i]
                cells.append(pct(q["accuracy"]) + (f" ± {q['accuracy_std'] * 100:.1f}" if q["model"] == "GRU" else ""))
            else:
                cells.append("[pending]")
        rows.append([r["model"], yn(r["velocity"]), yn(r["augment"]), yn(r["arms"])] + cells)
    table(rows, [1.6, 0.7, 0.7, 0.6, 1.2, 1.2])
    signs_figure(os.path.join(ROOT, "results_signs_20", "signs_bar.png"))
    figure("results_signs_20/signs_bar.png", 5.6, "Figure: word-sign accuracy on held-out participants (mean ± std over 3 seeds).")
    for S, n in ((S20, 20), (S50, 50)):
        if not S:
            continue
        R = S["rows"]
        best = max(R[1:], key=lambda r: r["accuracy"])
        P(f"{n} signs (chance {100 / n:.0f}%): the frame-average baseline reaches {pct(R[0]['accuracy'])}; the plain GRU "
          f"{pct(R[1]['accuracy'])} ({(R[1]['accuracy'] - R[0]['accuracy']) * 100:+.1f} points from modelling the order of frames). "
          f"Then velocity {(R[2]['accuracy'] - R[1]['accuracy']) * 100:+.1f}, augmentation {(R[3]['accuracy'] - R[2]['accuracy']) * 100:+.1f} "
          f"and arms {(R[4]['accuracy'] - R[3]['accuracy']) * 100:+.1f} points, ending at {pct(best['accuracy'])} (top-5 "
          f"{pct(best['top5'])}). The best model without arms, which is what the demo can run, reaches {pct(R[3]['accuracy'])}. "
          "Hardest signs: " + ", ".join(f"{s} ({a * 100:.0f}%)" for s, a in R[-1]["worst"][:4]) +
          "; most common confusions: " + ", ".join(f"{p}" for p, c in R[-1]["confusions"][:3]) + ".")
    P("[Takeaway in your own words: which addition mattered, by how much, and why you think so.]")
    P("The live demo only runs MediaPipe Hands, not the pose model, so it uses the best configuration without arms "
      "(model_gru_demo.pt). We have not yet tested the word model on our own signing; that happens at filming.")
else:
    P("[PENDING: train_signs.py --grid results]")

# ============================================================================ iteration 8
H("Iteration 8: from letter predictions to a caption")
if CAP:
    R = CAP["landmarks"]["rules"]
    signers = list(R["raw"])
    P("The letter classifier gives one prediction per webcam frame, so turning it into "
      "text is its own problem. Each of us fingerspelled the same 10 words (HELLO, BOOK, PENN, CAT, DOG, NAME, WATER, FIVE, "
      "TAXI, GOOD) with record_words.py, dropping the hand out of view after each word. The letter model for each person "
      "is trained on the Kaggle photos plus the other person's alphabet only. We score character error rate (CER: edits "
      "needed divided by word length) per word, and on the stream of all 10 words joined, where missing or extra spaces "
      "also count. Each rule's settings are tuned on one person and tested on the other.")
    names = {"raw": "1. Every frame (argmax, repeats collapsed)", "vote": "2. Majority vote over W frames (midterm demo)",
             "weighted": "3. Confidence-weighted vote", "hold": "4. + hold rule", "space": "5. + space when the hand drops"}
    rows = [["Rule"] + [f"{s} word CER" for s in signers] + ["Mean word CER", "Mean stream CER"]]
    for k in names:
        rows.append([names[k]] + [pct(R[k][s]["word_cer"]) for s in signers] +
                    [pct(np.mean([R[k][s]["word_cer"] for s in signers])), pct(np.mean([R[k][s]["stream_cer"] for s in signers]))])
    table(rows, [2.6, 0.9, 0.9, 1.0, 1.1])
    hold = R["space"][signers[0]]["params"]
    P(f"Writing a letter every time the prediction changes produces hundreds of percent CER: the hand passes through "
      f"other letter shapes on the way between letters. Voting helps, weighting votes by confidence helps more, and the "
      f"hold rule (a letter must stay on top for H frames; the same letter is written twice only if the wrist moved "
      f"sideways between the two holds) gives the biggest drop. The space rule does not change single-word CER but fixes "
      f"word breaks in the stream. Chosen settings when testing {signers[0]} (tuned on {signers[1]}): window "
      f"{hold['W']}, hold {hold['H']} frames, repeat after {hold['d_min']} hand-lengths of movement, space after "
      f"{hold['G']} frames without a hand.")
    ex = R["space"]
    rows = [["Word"] + [f"{s} caption" for s in signers]]
    for w in ex[signers[0]]["per_word"]:
        rows.append([w] + [ex[s]["per_word"].get(w, "") for s in signers])
    table(rows, [1.2, 1.6, 1.6])
    P("[Takeaway: where the remaining errors come from. Most are letter swaps between similar hand shapes "
      "(see the confusion pairs in iteration 9) and missed double letters, not timing.]")

# ============================================================================ iteration 9
H("Iteration 9: two-stage landmarks on real webcam data, and the final letter model")
if FIN:
    rows_ = FIN["rows"]
    get = lambda setup, lm: rows_[f"{setup} | {lm}"]
    P("In iteration 6 a simulation (shrinking Kaggle hands inside a 1080p frame) showed that detecting the hand, cropping "
      "around it and running MediaPipe again (two-stage) keeps accuracy up when the hand is small. Eva re-recorded the "
      "alphabet with the v2 recorder, which saves both landmark sets from the same frames, so the extractor is the only "
      "thing that changes between the rows below. Test: Eva's new recording (eva_close). 'Unseen signer' trains on Kaggle "
      "plus Isha only; 'new session' also adds Eva's older recording.")
    rows = [["Train", "Landmarks", "Accuracy"] + [f"size {b}" for b in FIN["bins"]]]
    for r in rows_.values():
        rows.append([r["setup"], r["landmarks"], pct(r["accuracy"])] + [pct(a) for a in r["by_hand_height"]])
    table(rows, [1.0, 0.9, 0.8, 0.75, 0.85, 0.85, 0.75])
    figure("results_final/hand_size_acc.png", 4.8, "Figure: letter accuracy on Eva by hand size, single pass vs two-stage.")
    u1, u2 = get("unseen signer", "single pass"), get("unseen signer", "two-stage")
    P(f"On real data two-stage did not help overall ({pct(u2['accuracy'])} vs {pct(u1['accuracy'])} for an unseen signer). "
      f"It helps on the smallest hands in the recording ({pct(u2['by_hand_height'][0])} vs {pct(u1['by_hand_height'][0])}) "
      f"and hurts on large ones. Two likely reasons: this recording was made close to the camera (hand size mostly "
      f"0.45 to 0.75 of the frame height), so it rarely reaches the regime where the simulation showed a gain; and every "
      f"training source (Kaggle photos, the older recordings) is single pass, so two-stage test landmarks are slightly "
      f"different from what the model was trained on. Adding Eva's older session raises accuracy by about "
      f"{(get('new session', 'single pass')['accuracy'] - u1['accuracy']) * 100:.0f} points, more than either extractor change.")
    P("Most confused pairs: " + ", ".join(f"{p[0]}/{p[1]}" for p, c in FIN["confusable_pairs"][:8]) +
      ". G/P and K/V differ mainly in which way the hand points, R/U/V in finger crossing and spread, and M/S/T in where "
      "the thumb sits under the fingers, which xy landmarks capture poorly.")
    P(f"Final letter model (results_final/model_final.joblib): Kaggle photos plus all three of our recordings (capped at "
      f"40 frames per letter per recording), tilt and jitter augmentation, logistic regression on normalized xy, "
      f"{FIN['final_n_train']:,} training samples after augmentation.")
    P("[Decision for the demo: keep two-stage only when the hand is small, or switch the demo to single pass?]")

# ============================================================================ iteration 10
H("Iteration 10: cleaning up the caption with a dictionary or an LLM")
P("The raw caption still has letter swaps (iteration 8). We compare two fixes on the 20 recorded words: a dictionary "
  "baseline that replaces each word with the closest of the 20,000 most common English words, where swapping two "
  "often-confused letters costs half as much, and an LLM given the raw caption and the same confused-letter pairs. "
  "Over-correction is how often a word that was already right gets changed.")
if CLEAN:
    rows = [["Version", "CER", "Word accuracy", "Over-correction", "Made worse", "Latency / call"]]
    first = next(iter(CLEAN.values()))
    t = first["table"]
    rows.append(["Raw caption", pct(t["raw"]["all"]["cer"]), pct(t["raw"]["all"]["word_acc"]), "-", "-", "-"])
    rows.append(["Dictionary", pct(t["dictionary"]["all"]["cer"]), pct(t["dictionary"]["all"]["word_acc"]),
                 pct(t["dictionary"]["all"]["over_correction"]), pct(t["dictionary"]["all"]["made_worse"]), "< 0.1 s"])
    llm_rows = [d for d in CLEAN.values() if "llm" in d["table"]]
    for d in llm_rows:
        q = d["table"]["llm"]["all"]
        rows.append([f"LLM: {d['model']}", pct(q["cer"]), pct(q["word_acc"]), pct(q["over_correction"]),
                     pct(q["made_worse"]), f"{np.mean(d['latencies']):.2f} s"])
    if not llm_rows:
        rows.append(["LLM (API)", "[Eva]", "", "", "", ""])
        rows.append(["LLM (Ollama, local)", "[Eva]", "", "", "", ""])
    table(rows, [1.6, 0.7, 1.0, 1.1, 0.9, 1.0])
    P(f"The dictionary fix only helps when the raw caption is close to a real word; with the raw CER around "
      f"{pct(t['raw']['all']['cer'])}, many captions are closer to some other common word (GOOD read as VOX becomes BOX). "
      "[Eva: one sentence on the LLM result and which model goes in the demo, and why.]")

# ============================================================================ lessons / future work
H("Lessons learned [draft, rewrite in your own words]")
B("A model that scores 91% on a held-out Kaggle signer scored 56% on Eva's webcam and 45% on Isha's (iteration 5). "
  "Testing on people and cameras the model has never seen is the only number that predicts the demo.")
B("Data from the right person mattered more than modelling tricks: adding one earlier session of Eva raised accuracy on "
  "her new recording by about 8 points (iteration 9), more than augmentation (about +4 in iteration 5) or the two-stage "
  "extractor (about -2).")
B("A result from simulation (two-stage on shrunken hands) did not carry over to real recordings in the form we expected, "
  "and our first way of binning by hand size (hand height) counted sideways letters such as G and P as small hands, which "
  "flipped the small-hand result. Check what a metric actually measures.")
B("Turning frame predictions into a caption is a separate problem from classifying frames: the hold rule cut character "
  "error rate by more than any change to the classifier.")

H("Future work [draft]")
B("Train a sequence model directly on fingerspelling video (e.g. ChicagoFSWild) with CTC loss, so letters and timing are "
  "learned together instead of frame classifier plus rules.")
B("Record more people under more lighting and camera conditions; two signers cannot show how the system generalizes.")
B("Combine word signs and fingerspelling in one stream with a switch that decides which model to trust.")
B("Use depth or both hands for the letter pairs xy landmarks cannot separate (M/N/S/T).")

doc.save(OUT)
print("wrote", OUT)
