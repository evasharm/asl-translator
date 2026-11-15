"""Build Final_Project_Proposal.docx from the files in results/ (12 pt Times New Roman)."""
import json
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Final_Project_Proposal.docx")
RES = os.path.join(ROOT, "results")
J = lambda n: json.load(open(os.path.join(RES, n)))
R, R0, E, E0, D, RS = J("results.json"), J("results_nopad.json"), J("extraction_stats.json"), \
    J("extraction_stats_nopad.json"), J("demo_stats.json"), J("random_split.json")

b0 = R0["baseline"]   # no padding, raw coords, logistic regression
b1 = R["baseline"]    # padding, raw coords, logistic regression
b2 = R["iter1"]       # padding, normalized, logistic regression
b3 = R["iter2"]       # padding, normalized + distances, MLP
pct = lambda x: f"{x * 100:.1f}%"
n_cls = len(E["per_class_detection"])
FONT = "Times New Roman"

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.6)
    s.left_margin = s.right_margin = Inches(0.75)


def set_font(run, size=12, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(k), FONT)


st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(3)
st.paragraph_format.line_spacing = 1.0


def H(text):
    p = doc.add_paragraph()
    set_font(p.add_run(text), 12, bold=True)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
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
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(path, width, caption):
    if not os.path.exists(path):
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
set_font(p.add_run("CIS 5810 Final Project Proposal, Group 9, Option 1\n"), 12)
set_font(p.add_run("Isha Bhatia (ishabh@engineering.upenn.edu), Eva Sharma (eshar28@engineering.upenn.edu)"), 12)

# ---------------- 1 goals ----------------
H("1. Goals and audience")
P("Our project will be building a vision model that translates sign language into captions in real time. We will use an "
  "existing hand tracking and detection model as our base for extracting hand signals first. We will then use labeled "
  "datasets of ASL signs to train a model that receives either image snapshots or pixel coordinates of important hand "
  "landmarks as input, and can identify the translations to form word and phrase captions.")
P("Our audience is hearing people who do not know ASL and want to follow a Deaf or hard of hearing signer when no "
  "interpreter is around, for example at a front desk, in a classroom, or on a video call. It is also for Deaf users who "
  "want a simple tool that runs on a normal laptop webcam with no special hardware.")
P("Our goals in order are: (1) recognize the static ASL alphabet and digits from single frames in real time, which is what "
  "this proposal tests, (2) recognize single word signs from short clips, which needs motion over several frames, and "
  "(3) turn the predictions into stable captions on screen. Full sentence translation and ASL grammar are out of scope "
  "for this semester.")

# ---------------- 2 pipeline ----------------
H("2. Pipeline design")
P("Our main design choice is to not train a vision model on raw pixels. Instead we use MediaPipe HandLandmarker, an "
  "existing hand detection model, to find the hand and give us 21 landmark points on it. We only train a small classifier "
  "on top of those points. Training then takes seconds on a laptop, the classifier does not care about background or lighting, "
  "and the same per frame vector can later feed a sequence model for word level signs. Each stage is its own script that "
  "saves its output to a file, so we never rerun MediaPipe while trying new features or models.")
figure(os.path.join(RES, "pipeline.png"), 6.2, "Figure 1. Our pipeline. The blue stages are existing tools, the green and orange stages are ours.")
B("OpenCV reads a webcam, a video file, or an image (hand_landmarks_demo.py).", "Capture.")
B(f"MediaPipe returns 21 (x, y, z) points per hand plus a left or right label. We pack this into one 126 number vector per "
  f"frame (left hand slot, right hand slot, zeros if a hand is missing). On our own {D['n_frames']} frame 720p webcam clip it "
  f"runs at about {D['model_fps']:.0f} FPS on a laptop CPU, see Figure 2 left (extract_landmarks.py).", "Landmark extraction.")
B("We move each hand so the wrist is at the origin, scale it so the wrist to middle knuckle distance is 1, and mirror left "
  "hands so every sample looks like a right hand. This removes where the hand is, how far it is from the camera, and which "
  "hand it is (features.py).", "Normalization.")
B(f"A classifier maps the vector to one of {n_cls} classes (A to Z and 0 to 9). We compare logistic regression and a small MLP. "
  f"For word level signs this block will become a GRU or temporal CNN over a sequence of frames (train_baseline.py).",
  "Classification.")
B("We take a majority vote over the last 8 frames and only add a letter to the caption when it wins at least half of them, "
  "so the caption does not flicker (demo_predict.py).", "Captioning.")

# ---------------- 3 baseline ----------------
H("3. Baseline results")
P(f"We used the Kaggle ASL Dataset (Rajput, 2020). It has {E['n_images']:,} photos of hands cut out on a black background, "
  f"{n_cls} classes (0 to 9 and A to Z), about 70 images per class, from five different people (hand1 to hand5 in the file "
  f"names). Since the file names tell us which person is in each image, we test on a person the model has never seen: all "
  f"{b1['n_test_total']} images of hand2 are the test set and the other four people ({b1['n_train'] + b1['n_test_total'] - b1['n_test']:,} "
  f"images) are the training set. We first tried a 900 image Kaggle set, but it already had MediaPipe's skeleton drawn on every "
  f"picture, so MediaPipe could not find a hand in it.", "Data.")
P(f"We ran MediaPipe on the raw images, used the raw 63 coordinates as features, and trained logistic regression. MediaPipe only "
  f"found a hand in {pct(E0['det_rate'])} of the images ({E0['n_detected']:,} of {E0['n_images']:,}). The failures were not random. "
  f"Closed fist letters almost never got detected ({', '.join(f'{c.upper()} {r * 100:.0f}%' for c, r in E0['worst_detection'][:5])}) "
  f"while open hand shapes like 5 and 9 were found every time. On the test images where a hand was found, the classifier got "
  f"{pct(b0['accuracy'])} ({pct(b0['top3'])} top 3). If we count an image with no detected hand as wrong, the strict accuracy is "
  f"{pct(b0['accuracy_strict'])}. We report both numbers from here on.", "Baseline.")

# ---------------- 4 iterations ----------------
H("4. Development iterations from the baseline")
P(f"When we looked at the failed images we saw that the hand fills the whole frame, and MediaPipe's palm detector needs some "
  f"space around the hand. We added a black border of 50% of the image size and mapped the points back to the original image. "
  f"Detection went from {pct(E0['det_rate'])} to {pct(E['det_rate'])} with no change to the model (Figure 2). With the same "
  f"features and classifier, strict accuracy went from {pct(b0['accuracy_strict'])} to {pct(b1['accuracy_strict'])}.",
  "Iteration 1: add padding so MediaPipe finds the hand.")
figure(os.path.join(RES, "figure_detection.jpg"), 4.6,
       "Figure 2. Left: landmarks on our webcam clip. Middle and right: the same 'S' image before and after padding. "
       "Without padding no hand is found.")
P(f"Centering on the wrist, scaling by hand size, and mirroring raised accuracy from {pct(b1['accuracy'])} to "
  f"{pct(b2['accuracy'])} ({pct(b2['accuracy_strict'])} strict). This only helps when testing on a new person. On a random "
  f"80/20 split where the same people are in train and test, raw coordinates already get {pct(RS['raw'])} and normalization "
  f"does not help ({pct(RS['normalized'])}). So normalization is removing person specific hand size and position cues, "
  f"which is exactly what we want for new users.", "Iteration 2: normalize the landmarks.")
P(f"Adding the 210 distances between every pair of points and switching to a 2 layer MLP (256, 128) gave {pct(b3['accuracy'])}, "
  f"a bit below logistic regression. With only {b3['n_train']:,} training samples the bigger model does not help yet, so "
  f"logistic regression on normalized landmarks is our current best.", "Iteration 3: bigger features and an MLP.")
rows = [["Run", "Padding", "Features", "Model", "Hand found", "Test acc", "Strict", "Merged"],
        ["Baseline", "none", "raw (63)", "logistic reg.", pct(E0["det_rate"]), pct(b0["accuracy"]), pct(b0["accuracy_strict"]), pct(b0["accuracy_merged"])],
        ["Iteration 1", "50%", "raw (63)", "logistic reg.", pct(E["det_rate"]), pct(b1["accuracy"]), pct(b1["accuracy_strict"]), pct(b1["accuracy_merged"])],
        ["Iteration 2", "50%", "normalized (63)", "logistic reg.", pct(E["det_rate"]), pct(b2["accuracy"]), pct(b2["accuracy_strict"]), pct(b2["accuracy_merged"])],
        ["Iteration 3", "50%", "norm. + dist. (273)", "MLP (256, 128)", pct(E["det_rate"]), pct(b3["accuracy"]), pct(b3["accuracy_strict"]), pct(b3["accuracy_merged"])]]
table(rows, [1.0, 0.7, 1.4, 1.15, 0.85, 0.8, 0.7, 0.75])
P(f"All rows are tested on hand2 ({b1['n_test_total']} images). Test acc is over images where a hand was found. Strict counts an "
  f"undetected image as wrong. Merged counts 0/O, 2/V, and 6/W as the same class. Those three pairs make up most of the "
  f"remaining errors, and in ASL they really are the same hand shape, so no single frame model can tell them apart. We will "
  f"need context from neighboring letters to do that. The next most confused pairs, U/R and T/S, differ only in thumb "
  f"position, which one camera often cannot see. Confusion matrices are in results/ in our repo.")

# ---------------- 5 plan ----------------
H("5. Plan for improvements")
B("Record about 30 seconds per letter from each of us with hand_landmarks_demo.py on a real background, and test training "
  "on one of us and testing on the other. We expect a drop from the clean dataset. We will try to close it with landmark "
  "augmentation (random rotation, scale, jitter).", "Our own data (iteration 4).")
B("Google's Kaggle ASL Signs set already comes with MediaPipe landmarks for 250 word signs from 21 people, so no extraction "
  "is needed and we can split by person. We will resample each clip to 32 frames, add frame to frame velocity, and train "
  "a GRU or 1D temporal CNN, starting with 20 signs.", "Word level signs (iteration 5).")
B("Handle J and Z, which involve motion, with the same sequence model, add pose landmarks for two handed signs, and check "
  "whether the z coordinate helps with the thumb confusions.", "Motion letters and pose (iteration 6).")
B("Weight the vote by confidence, require a letter to be held for a short time before adding it, use context to resolve "
  "0/O type ties, and insert spaces between words. Target 25 FPS end to end on a laptop.", "Real time captions.")

# ---------------- 6 timeline ----------------
H("6. Timeline, milestones, and member duties")
rows = [["Dates", "Milestone", "Isha", "Eva"],
        ["10/12 to 10/18", "TA meeting, record our own data, test on real backgrounds (iter 4)", "Augmentation, evaluation code", "Recording setup, data collection, failure analysis"],
        ["10/19 to 11/1", "Load ASL Signs landmarks, GRU / temporal CNN on 20 signs (iter 5)", "Sequence model, training loop", "Clip loader, resampling, velocity features"],
        ["11/2 to 11/15", "Scale to 100+ signs, pose landmarks, J and Z (iter 6)", "Model scaling, ablations", "Pose integration, error analysis"],
        ["11/16 to 11/20", "Midterm report", "Pipeline and results sections", "Data and iteration sections"],
        ["11/21 to 12/5", "Live caption demo, commit rules, demo video", "Captioning logic, latency", "Demo UI, video, slides"],
        ["12/6 and 12/10", "Final Project Day, final report", "Final report", "Final report"]]
table(rows, [1.0, 2.8, 1.5, 1.7])
P("We both do the literature search and keep a shared experiment log.")

H("7. TA request (optional)")
P("A TA who has worked with sequence models on landmark or pose data would help us check the word level design before we "
  "commit to it.")

H("Use of AI tools and attribution")
P("We used Claude (Anthropic) to help write the extraction, feature, and training scripts and to format this document. We "
  "reviewed and ran all of the code ourselves. MediaPipe HandLandmarker (Apache 2.0) is used as is. Dataset: A. Rajput, "
  "ASL Dataset, Kaggle, 2020.")

doc.save(OUT)
print("wrote", OUT)
