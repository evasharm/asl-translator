"""Build Final_Project_Midterm_Report.docx (12 pt Times New Roman, same style as the proposal).

Reads results/ (proposal numbers), results_aug/ (iteration 4) and, when present,
results_own/ (iteration 5) and results_caption/ (iteration 6). Sections whose
numbers are not there yet are written with [PLACEHOLDER] markers.
"""
import json
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Final_Project_Midterm_Report.docx")
J = lambda *p: json.load(open(os.path.join(ROOT, *p))) if os.path.exists(os.path.join(ROOT, *p)) else None

R = J("results", "results.json")
R0 = J("results", "results_nopad.json")
E = J("results", "extraction_stats.json")
E0 = J("results", "extraction_stats_nopad.json")
RS = J("results", "random_split.json")
A = J("results_aug", "results.json")
OWN = J("results_own", "results.json")
CAP = J("results_caption", "results.json")

b0, b1, b2, b3 = R0["baseline"], R["baseline"], R["iter1"], R["iter2"]
pct = lambda x: f"{x * 100:.1f}%"
a = lambda k: A[f"{k}/logreg"]
am = lambda k: A[f"{k}/mlp"]
FONT = "Times New Roman"

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


def table(rows, widths, size=10, bold_row=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            para = cell.paragraphs[0]
            para.paragraph_format.space_after = Pt(0)
            set_font(para.add_run(str(val)), size, bold=(r == 0 or r == bold_row))
            cell.width = Inches(widths[c])
            para.paragraph_format.keep_with_next = (r == 0)  # keep the header with the first row, let long tables break
        trpr = t.rows[r]._tr.get_or_add_trPr()
        trpr.append(trpr.makeelement(qn("w:cantSplit"), {}))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(path, width, caption):
    if not os.path.exists(path):
        P(f"[FIGURE PLACEHOLDER: {os.path.basename(path)} not generated yet]")
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(c.add_run(caption), 10, italic=True)


# ---------------- header ----------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("Real-Time ASL Captions"), 14, bold=True)
p.paragraph_format.space_after = Pt(0)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(p.add_run("CIS 5810 Final Project Midterm Report, Group 9, Option 1\n"), 12)
set_font(p.add_run("Isha Bhatia (ishabh@engineering.upenn.edu), Eva Sharma (eshar28@engineering.upenn.edu)"), 12)

# ---------------- 1 summary ----------------
X = J("results_aug_xy", "results.json")
OWN = J("results_own", "results.json")
HS = J("results_own", "hand_size.json")
TS = J("results_twostage", "results.json")
x = lambda k: X[f"{k}/logreg"]
own = lambda k: OWN[k]
H("1. Summary")
P(f"We are building a system that watches a signer through a laptop webcam and shows live captions of American Sign "
  f"Language. We use MediaPipe to find the hand and give us 21 landmark points, and we train a small classifier on top of "
  f"those points. In the proposal we showed a baseline that recognized the static ASL alphabet and digits on a person the "
  f"model had never seen with {pct(b0['accuracy_strict'])} strict accuracy, and three iterations (padding the image so MediaPipe "
  f"finds the hand, normalizing the landmarks, and trying an MLP) that raised it to {pct(b2['accuracy_strict'])}.")
P(f"Since the proposal we have done three more iterations. Iteration 4 augments the training landmarks with synthetic "
  f"copies and raises held-out accuracy from {pct(x('none')['accuracy'])} to {pct(x('jitter')['accuracy'])}. Iteration 5 is the "
  f"important one: we each recorded the full alphabet on our own laptop webcams in our rooms, and the model that scores 91% "
  f"on the dataset scored only {pct(own('kaggle -> isha')['accuracy'])} and {pct(own('kaggle -> Eva')['accuracy'])} on us. We traced "
  f"that to three causes (the hand being small in the frame, MediaPipe's depth value being unreliable, and a scaling bug in our "
  f"own extraction code), fixed them, and showed that adding a small amount of the other person's webcam data brings a new "
  f"user to {pct(own('kaggle+other -> isha')['accuracy'])} and {pct(own('kaggle+other+aug -> Eva')['accuracy'])}. Iteration 6 changes "
  f"the pipeline so MediaPipe runs a second time on a crop around the hand, which keeps accuracy at 86% to 89% as the hand "
  f"shrinks in the frame where the single pass falls to 70%. Word level signs and the captioning rules move to the final report.")

# ---------------- 2 goals ----------------
H("2. Goals and audience")
P("Our project is building a vision model that translates sign language into captions in real time. We use an existing "
  "hand tracking and detection model as our base for extracting hand signals first, then use labeled datasets of ASL signs "
  "to train a model that receives pixel coordinates of important hand landmarks as input and identifies the sign to form "
  "word and phrase captions.")
P("Our audience is hearing people who do not know ASL and want to follow a Deaf or hard of hearing signer when no "
  "interpreter is around, for example at a front desk, in a classroom, or on a video call. It is also for Deaf users who "
  "want a simple tool that runs on a normal laptop webcam with no special hardware.")
P("Our goals in order are: (1) recognize the static ASL alphabet and digits from single frames in real time, (2) make "
  "that work for new people on real webcam video, not only on the clean dataset, (3) turn the predictions into stable "
  "captions on screen, and (4) recognize single word signs from short clips, which needs motion over several frames. "
  "Goals 1 to 3 are the focus of this report. Full sentence translation and ASL grammar are out of scope for this semester.")

# ---------------- 3 pipeline ----------------
H("3. Pipeline")
P("Our main design choice is to not train a vision model on raw pixels. Instead we use MediaPipe HandLandmarker, an "
  "existing hand detection model, to find the hand and give us 21 landmark points on it, and we only train a small "
  "classifier on top of those points. Training takes seconds on a laptop, the classifier does not care about background or "
  "lighting, and the same per frame vector can later feed a sequence model for word level signs. Each stage is its own "
  "script that saves its output to a file, so we never rerun MediaPipe while trying new features or models. Figure 1 shows "
  "the pipeline as it stands now, with the augmentation stage added in iteration 4 and the two-stage landmark extraction from iteration 6.")
figure(os.path.join(ROOT, "results_aug", "pipeline_midterm.png"), 7.0,
       "Figure 1. Our pipeline. Blue stages are existing tools, green and orange stages are ours.")
B("OpenCV reads a webcam, a video file, or an image (hand_landmarks_demo.py, record_signs.py).", "Capture.")
B("MediaPipe returns 21 (x, y, z) points per hand plus a left or right label. For dataset images we add a black border of "
  "50% before detection (proposal iteration 1) and map the points back. For webcam frames we now run MediaPipe twice: once "
  "on the full frame to find the hand, then again on a tight crop around it (iteration 6, twostage.py). Webcam x values are "
  "rescaled into frame-height units so a 16:9 frame matches the square dataset images (extract_landmarks.py, record_signs.py).",
  "Landmark extraction.")
B("We move each hand so the wrist is at the origin, scale it so the wrist to middle knuckle distance is 1, and mirror left "
  "hands so every sample looks like a right hand. Since iteration 5 we drop the z (depth) value and use only x and y, 42 "
  "numbers per frame (features.py).", "Normalization.")
B("During training only, we make 5 extra copies of every training hand with small random noise on each point and a random "
  "3D tilt. This is new in iteration 4 (augment.py).", "Augmentation.")
B("Logistic regression maps the 42 normalized values to one of 36 classes (A to Z and 0 to 9). We also tried an MLP "
  "and it has not beaten logistic regression on any of our tests (train_baseline.py, train_augment.py, train_own.py).",
  "Classification.")
B("We take a majority vote over the last 8 frames and only add a letter to the caption when it wins at least half of them "
  "(demo_predict.py). A confidence weighted vote, a hold rule, automatic spaces, and a language model cleanup of the caption "
  "text are planned for the final version.", "Captioning.")

# ---------------- related work ----------------
H("4. Related work")
P("Our design follows the pose based branch of sign language recognition, where an off the shelf keypoint detector "
  "replaces a learned image model and only a small classifier is trained on the keypoints. SPOTER (Bohacek and Hruz, "
  "WACVW 2022) feeds MediaPipe style body and hand keypoints to a small transformer for word level signs and reports that "
  "normalizing the keypoints was by far the largest gain on WLASL100 (45% to 59%), with each keypoint augmentation adding "
  "1 to 2 points on top. That is the same order of effect we saw: normalization in iteration 2 was the big step and "
  "augmentation in iteration 4 the small one. Sanalohit and Katanyukul (2022) use MediaPipe Hands landmarks with a small "
  "fully connected network for Thai fingerspelling and find that wrist relative coordinates beat absolute coordinates "
  "(84.6% vs 82.1%), which matches our iteration 2 on a different alphabet.")
P("For augmentation, SPOTER uses in plane rotation up to 13 degrees, a perspective tilt, horizontal squeeze and Gaussian "
  "noise on the keypoints. Our 3D tilt and jitter are their perspective and noise augmentations. Rotation helped them on "
  "whole body word signs but hurt us on fingerspelling, which we explain in iteration 4 by the fact that several "
  "fingerspelled letters are defined partly by hand orientation. Nakamura and Jing (IEEE Access 2025) go further and "
  "generate adversarial skeleton samples, which we have not tried.")
P("For evaluation we follow ChicagoFSWild (Shi et al., 2018), which assigns every signer to only one split so that test "
  "accuracy reflects a person the model never saw. Their best letter accuracy on fingerspelling in real online video was "
  "about 42%, against roughly 83% for humans, so our numbers on static, isolated, clean background images should not be "
  "compared with theirs directly. They also note that A, S, T and N differ only by thumb position and that R, U and V all "
  "extend the index and middle fingers, and their most common error was U read as R. Our confusion pairs (U to R, T to S) "
  "are the same ones, which suggests they are a property of single camera fingerspelling rather than of our classifier.")

# ---------------- 4 baseline ----------------
H("5. Baseline results")
P(f"Kaggle ASL Dataset (Rajput, 2020): {E['n_images']:,} photos of hands cut out on a black background, 36 classes, about 70 "
  f"per class, from five different people (hand1 to hand5 in the file names). We always test on a person the model has never "
  f"seen: all {b1['n_test_total']} images of hand2 are the test set and the other four people ({b1['n_train'] + b1['n_test_total'] - b1['n_test']:,} "
  f"images) are the training set.", "Data.")
P(f"MediaPipe on the raw images, raw 63 coordinates, logistic regression. MediaPipe only found a hand in {pct(E0['det_rate'])} "
  f"of images, mostly missing closed fist letters (T {E0['per_class_detection']['t'] * 100:.0f}%, M {E0['per_class_detection']['m'] * 100:.0f}%, "
  f"S {E0['per_class_detection']['s'] * 100:.0f}%). On the test images where a hand was found the classifier got {pct(b0['accuracy'])}. "
  f"Counting an image with no detected hand as wrong, the strict accuracy is {pct(b0['accuracy_strict'])}. We report both numbers, "
  f"and a merged accuracy that counts 0/O, 2/V and 6/W as the same class because they are the same handshape in ASL.", "Baseline.")
P(f"Iteration 1 added a 50% black border so MediaPipe finds the hand (detection {pct(E0['det_rate'])} to {pct(E['det_rate'])}, "
  f"strict accuracy {pct(b0['accuracy_strict'])} to {pct(b1['accuracy_strict'])}). Iteration 2 normalized the landmarks "
  f"({pct(b1['accuracy'])} to {pct(b2['accuracy'])}). Iteration 3 tried pairwise distances and an MLP ({pct(b3['accuracy'])}, "
  f"no gain). Iteration 2, logistic regression on normalized landmarks, is the starting point for everything below.",
  "Proposal iterations, for reference.")
rows = [["Run", "Padding", "Features", "Model", "Hand found", "Test acc", "Strict", "Merged"],
        ["Baseline", "none", "raw (63)", "logistic reg.", pct(E0["det_rate"]), pct(b0["accuracy"]), pct(b0["accuracy_strict"]), pct(b0["accuracy_merged"])],
        ["Iteration 1", "50%", "raw (63)", "logistic reg.", pct(E["det_rate"]), pct(b1["accuracy"]), pct(b1["accuracy_strict"]), pct(b1["accuracy_merged"])],
        ["Iteration 2", "50%", "normalized (63)", "logistic reg.", pct(E["det_rate"]), pct(b2["accuracy"]), pct(b2["accuracy_strict"]), pct(b2["accuracy_merged"])],
        ["Iteration 3", "50%", "norm. + dist. (273)", "MLP (256, 128)", pct(E["det_rate"]), pct(b3["accuracy"]), pct(b3["accuracy_strict"]), pct(b3["accuracy_merged"])]]
table(rows, [1.0, 0.7, 1.4, 1.15, 0.85, 0.8, 0.7, 0.75])

# ---------------- 5 iterations ----------------
H("6. Development iterations since the proposal")

H2("Iteration 4: landmark augmentation")
P("The model is tested on a person it never saw, so the question is whether we can make the training set look like more "
  "people without recording more people. We make 5 extra copies of every training hand with random changes applied around "
  "the wrist, and test each change on its own and together, over 3 random seeds. The four changes are: rotate the hand in "
  "the image plane by up to 15 degrees, scale it by 0.8 to 1.2, tilt it in 3D by up to 25 degrees about the vertical axis "
  "(using MediaPipe's z value), and add Gaussian noise of 1% of the hand size to every point. We also tried the opposite "
  "approach for rotation: rotate every hand in the features so the wrist to middle knuckle line always points up "
  "(rotation normalization). The table has two columns because of something we found in iteration 5: our extraction code "
  "had left z at half its correct scale in the cached dataset, and on webcam frames z turned out to be unreliable anyway. "
  "We rerun the grid with the corrected z (left) and with z dropped (right), which is what we use from here on.", "What we did.")
rows = [["Augmentation", "Train size", "With z (63 features)", "Without z (42 features)", "Strict, without z"]]
for name, key in [("none (proposal iteration 2)", "none"), ("rotate", "rotate"), ("scale", "scale"), ("3D tilt", "tilt"),
                  ("jitter", "jitter"), ("tilt + jitter", "tilt+jitter"), ("all four", "all"),
                  ("rotation normalized features", "rotnorm"), ("all four + rotation normalized", "all+rotnorm")]:
    xa = X.get(f"{key}/logreg")
    rows.append([name, f"{a(key)['n_train']:,}", pct(a(key)["accuracy"]), pct(xa["accuracy"]) if xa else "same as with z",
                 pct(xa["strict"]) if xa else "-"])
table(rows, [2.3, 0.9, 1.3, 1.5, 1.2], bold_row=5)
P("Logistic regression, test on hand2, mean of 3 seeds (standard deviation under 0.5 points in every row). The MLP was "
  "1 to 3 points below logistic regression in every row and is left out.", size=10)
figure(os.path.join(ROOT, "results_aug_xy", "augmentation_bar.png"), 5.0,
       "Figure 2. Accuracy on the held-out person for each augmentation, without z. The dashed line is no augmentation.")
P(f"With z dropped, jitter alone is the best single change ({pct(x('none')['accuracy'])} to {pct(x('jitter')['accuracy'])}), "
  f"tilt and jitter together are about the same ({pct(x('tilt+jitter')['accuracy'])}), and all four together are slightly worse "
  f"({pct(x('all')['accuracy'])}). Scale does nothing, as expected, because our normalization already removes hand size. "
  f"Using more than 5 copies per sample did not help. The gain on the dataset is small, about 1.5 points, but on our own "
  f"webcam recordings in iteration 5 the same augmentation adds {(own('kaggle+aug -> isha')['accuracy'] - own('kaggle -> isha')['accuracy']) * 100:.1f} "
  f"and {(own('kaggle+aug -> Eva')['accuracy'] - own('kaggle -> Eva')['accuracy']) * 100:.1f} points, so it matters more where the test data looks "
  f"least like the training data.", "What we found.")
P(f"In our first run of this grid, before we found the z bug, rotation augmentation looked clearly harmful "
  f"({pct(a('rotate')['accuracy'])} vs {pct(a('none')['accuracy'])} with z) and we had an explanation ready: letters like G, H, P and Q are "
  f"told apart partly by which way the hand points, so rotation invariance deletes a real cue. With z dropped, rotation is "
  f"neutral ({pct(x('rotate')['accuracy'])}). The explanation may still be partly true, but the effect we measured was an artifact "
  f"of a bug in our own preprocessing. The lesson for us is to check the input representation before trusting an ablation, "
  f"and we now keep the raw landmark cache and the feature code under test.", "A result we had to take back.")

H2("Iteration 5: our own webcam recordings")
P(f"The dataset is hands cut out on a black background. The final demo runs on a laptop webcam in a normal room, so we "
  f"wrote a recording script (record_signs.py) that shows one sign at a time, counts down, and saves MediaPipe landmarks for "
  f"15 seconds while the person holds the sign and moves the hand slowly. Each of us recorded all 36 signs in our own room "
  f"(Isha: 16,063 frames, Eva: 5,300 frames). No hand labeling was needed. We then ran five transfer tests for each of "
  f"us, where the test person is never in the training set: train on the dataset only, dataset with augmentation, the other "
  f"person only, dataset plus the other person, and dataset plus the other person with augmentation. Because 15,000 "
  f"near-duplicate webcam frames would swamp 1,500 dataset photos, we use at most 40 evenly spaced frames per sign from the "
  f"other person for training. Every number below is on a person the model never saw.", "What we did.")
rows = [["Train on", "Test on Eva", "Merged", "Test on Isha", "Merged", "Kaggle hand2"]]
for name, ke, ki in [("dataset only", "kaggle -> Eva", "kaggle -> isha"), ("dataset + augmentation", "kaggle+aug -> Eva", "kaggle+aug -> isha"),
                     ("other person only (40 frames/sign)", "other -> Eva", "other -> isha"),
                     ("dataset + other person", "kaggle+other -> Eva", "kaggle+other -> isha"),
                     ("dataset + other person + augmentation", "kaggle+other+aug -> Eva", "kaggle+other+aug -> isha")]:
    e, i = own(ke), own(ki)
    rows.append([name, pct(e["accuracy"]), pct(e["merged"]), pct(i["accuracy"]), pct(i["merged"]), f"{pct(e['kaggle_hand2_accuracy'])} / {pct(i['kaggle_hand2_accuracy'])}"])
table(rows, [2.3, 0.95, 0.8, 0.95, 0.8, 1.2], bold_row=4)
P("Logistic regression on x, y landmarks, mean of 2 seeds. Merged counts 0/O, 2/V, 6/W and 9/F as the same class because "
  "they are the same handshape in ASL. The last column is the same model tested back on the dataset's held-out person.", size=10)
figure(os.path.join(ROOT, "results_own", "snapshots.jpg"), 7.0,
       "Figure 3. The same three signs from Isha's recording, Eva's recording, and the dataset.")
P(f"The drop is large. The model that gets 91% on a new dataset person gets {pct(own('kaggle -> isha')['accuracy'])} on Isha and "
  f"{pct(own('kaggle -> Eva')['accuracy'])} on Eva. We found three reasons. First, hand size: Isha's hand was about "
  f"{HS['isha']['median_size']:.2f} of the frame height and Eva's {HS['eva']['median_size']:.2f}, while the dataset hands fill the whole image. "
  f"Within each recording, accuracy rises steadily with hand size (Figure 4), from {HS['isha']['by_size'][0][3] * 100:.0f}% on Isha's smallest "
  f"hands to {max(r[3] for r in HS['isha']['by_size']) * 100:.0f}% on her largest, and Eva, who recorded closer, scores higher overall. "
  f"MediaPipe's landmarks on a small hand are noisy, and the thumb in particular is often read as tucked, which is why 3 was "
  f"read as V, and 4 and 5 as B. Second, depth: MediaPipe's z value is noisy on small hands and is defined differently between "
  f"library versions. Dropping it raised Isha from 35% to 45% and did not hurt on the dataset (91.3% to 92.1%). Third, a bug: "
  f"while tracing the gap we found that our extraction script, when undoing the 50% padding, rescaled x and y back to the "
  f"original image but not z, so every cached dataset landmark had z at half scale. We fixed the script and the cache.", "What we found.")
figure(os.path.join(ROOT, "results_own", "hand_size.png"), 4.6,
       "Figure 4. Accuracy of the dataset-trained model on our frames, by hand height in the frame.")
P(f"Adding the other person's webcam data helps more than anything else: Isha goes from {pct(own('kaggle -> isha')['accuracy'])} to "
  f"{pct(own('kaggle+other -> isha')['accuracy'])} with only 1,360 of Eva's frames added, and Eva from {pct(own('kaggle -> Eva')['accuracy'])} to "
  f"{pct(own('kaggle+other+aug -> Eva')['accuracy'])}. Training on the other person alone, without the dataset, is worse "
  f"({pct(own('other -> isha')['accuracy'])} and {pct(own('other -> Eva')['accuracy'])}), so the dataset still carries most of the "
  f"signal and our recordings mostly teach the model what a webcam hand looks like. The models trained with our data lose "
  f"little on the dataset's own held-out person (last column), so nothing is traded away.", None)
P(f"The remaining errors are dominated by the homograph pairs: on both of us the most common confusions are W read as 6, "
  f"V as 2, 9 as F, and 0 as O. These are the same handshape and no single frame classifier can separate them, so the merged "
  f"accuracy ({pct(own('kaggle+other -> isha')['merged'])} and {pct(own('kaggle+other+aug -> Eva')['merged'])}) is the fairer number "
  f"for the classifier, and separating them is a job for the captioning stage using context. After those, the errors are "
  f"U read as R, N as M, and C as O, the thumb and finger crossing pairs that ChicagoFSWild also reports as the hardest. "
  f"J and Z are motion letters and are wrong by design for a static model.", None)

H2("Iteration 6: two-stage landmark extraction for small hands")
P("Iteration 5 showed that landmark quality falls as the hand gets smaller in the frame, and the dataset hands fill the "
  "whole image. Rather than asking users to sit close to the camera, we changed the pipeline: MediaPipe runs on the full "
  "frame to find the hand, we cut a square crop around it with a 50% margin, and MediaPipe runs again on the crop, so the "
  "landmark model always sees a large hand like the training data. The crop's landmarks are mapped back to frame "
  "coordinates (twostage.py). The recorder and the live demo now use this, and the recorder warns on screen when the hand "
  "is below 35% of the frame height.", "What we did.")
P("To measure it without re-recording, we take the dataset's held-out person (hand2), paste each image into a black "
  "1920x1080 frame at a chosen height, and compare single pass against two-stage landmarks with the iteration 2 classifier. "
  "Hand height 0.83 is the original image size, 0.35 and 0.25 are what our own recordings looked like.", None)
rows = [["Hand height / frame", "Hand found", "Single pass accuracy", "Two-stage accuracy", "Gain"]]
for f, r in TS.items():
    rows.append([f, f"{r['single']['detection'] * 100:.0f}%", pct(r["single"]["accuracy"]), pct(r["twostage"]["accuracy"]),
                 f"{(r['twostage']['accuracy'] - r['single']['accuracy']) * 100:+.1f}"])
table(rows, [1.5, 1.1, 1.5, 1.5, 0.8])
P("Accuracy is over images where the first pass found a hand, 10 images per class per size. Detection is the same for both "
  "because the two-stage method starts from the single pass detection.", size=10)
figure(os.path.join(ROOT, "results_twostage", "small_hand.png"), 4.6,
       "Figure 5. Single pass vs two-stage landmarks as the hand shrinks in a 1080p frame.")
P(f"At full size the two methods are the same ({pct(TS['0.83']['single']['accuracy'])} vs {pct(TS['0.83']['twostage']['accuracy'])}). "
  f"As the hand shrinks, the single pass falls to {pct(TS['0.20']['single']['accuracy'])} while the two-stage result stays between "
  f"{pct(min(r['twostage']['accuracy'] for r in TS.values()))} and {pct(max(r['twostage']['accuracy'] for r in TS.values()))}, a gain of "
  f"{(TS['0.25']['twostage']['accuracy'] - TS['0.25']['single']['accuracy']) * 100:.0f} points at the hand sizes we actually recorded at. "
  f"The cost is a second MediaPipe call per frame, about 10 ms on a laptop CPU, which still leaves the demo above 25 FPS. "
  f"What the second pass cannot fix is detection: on a black background the palm detector misses most hands below 25% of "
  f"the frame, which is why the hand found column drops. On our real recordings detection was 98% to 99% even at small "
  f"sizes because video mode tracks the hand between frames, so in practice the detection limit matters less than this "
  f"table suggests.", "What we found.")
P("Both of our recordings were made with the single pass version of the recorder, so the two-stage result is from the "
  "controlled experiment only. The next recording each of us makes will store both versions of the landmarks for every "
  "frame, which gives the real-webcam number for the final report.", "Limitation.")

# ---------------- 6 timeline ----------------
H("7. Timeline, milestones, and member duties")
rows = [["Dates", "Milestone", "Isha", "Eva", "Status"],
        ["10/9", "Proposal with baseline and iterations 1 to 3", "Extraction, features, training", "Dataset, proposal edits", "done"],
        ["10/12 to 10/25", "First TA meeting", "", "", "[fill in date]"],
        ["11/15", "Iteration 4: augmentation grid", "Augmentation code, experiments", "Review of results", "done"],
        ["11/16 to 11/19", "Iteration 5: record our own data, transfer tests, z bug fix", "Recording script, transfer code, analysis", "Recording session, ASL Signs download", "done"],
        ["11/18", "Iteration 6: two-stage landmarks, small hand experiment", "twostage.py, simulation", "Review, recorder test", "done"],
        ["11/19 to 11/20", "Midterm report and TA check-in", "Sections 3 to 6", "Sections 1, 2, 7, final read", ""],
        ["11/21 to 11/29", "Re-record with the two-stage recorder; word level signs from the ASL Signs subset (GRU), 20 then 50 signs", "Sequence model, training loop", "ASL Signs subset, clip loader, recordings", ""],
        ["11/30 to 12/5", "Captioning rules and word test; J and Z; demo video and slides", "Captioning logic, latency", "Word recordings, demo UI, video, slides", ""],
        ["12/6 and 12/10", "Final Project Day, final report", "Final report", "Final report", ""]]
table(rows, [0.95, 2.4, 1.4, 1.5, 0.8], size=9.5)
P("We both do the literature search and keep a shared experiment log in the repository README. Work is split by pipeline "
  "stage so each of us owns a script and its saved output.")
P("Compared with the proposal timeline, we are behind on the word level sign model and the captioning rules. We spent the "
  "midterm period on our own webcam data instead, because iteration 5 showed the letter model did not work on a real webcam "
  "yet, and fixing that first seemed more important than adding words on top of a model that did not transfer. The Google "
  "ASL Signs dataset (40 GB) is now downloaded on Eva's laptop and a script to extract a small hands-only subset is written, so "
  "word level signs are the first item after the midterm.", "Change from the proposal.")

H("References")
for ref in [
    "M. Bohacek and M. Hruz. Sign Pose-based Transformer for Word-level Sign Language Recognition. WACV Workshops, 2022.",
    "B. Shi, A. Martinez Del Rio, J. Keane, J. Michaux, D. Brentari, G. Shakhnarovich and K. Livescu. American Sign Language "
    "Fingerspelling Recognition in the Wild. IEEE SLT, 2018 (ChicagoFSWild).",
    "B. Shi, A. Martinez Del Rio, J. Keane, D. Brentari, G. Shakhnarovich and K. Livescu. Fingerspelling Recognition in the Wild "
    "with Iterative Visual Attention. ICCV, 2019.",
    "T. Sanalohit and T. Katanyukul. TFS Recognition: Investigating MPH, Thai Finger Spelling Recognition with MediaPipe Hands. "
    "arXiv:2201.03170, 2022.",
    "K. Nakamura and L. Jing. Skeleton-Based Data Augmentation for Sign Language Recognition Using Adversarial Learning. "
    "IEEE Access, vol. 13, 2025.",
    "F. Zhang et al. MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214, 2020.",
    "A. Rajput. ASL Dataset. Kaggle, 2020.",
]:
    P(ref, size=11)

H("Use of AI tools and attribution")
P("We used Claude (Anthropic) to help write the extraction, augmentation, recording, and training scripts and to format "
  "this document. We reviewed and ran all of the code ourselves and the experiment design and analysis are ours. MediaPipe "
  "HandLandmarker (Apache 2.0) is used as is.")

doc.save(OUT)
print("wrote", OUT)
