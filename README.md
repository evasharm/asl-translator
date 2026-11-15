# Real-Time ASL Captions (CIS 5810, Group 9)

Baseline: static ASL fingerspelling from MediaPipe hand landmarks.

## Setup

```bash
pip install -r requirements.txt
mkdir -p models data
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
kaggle datasets download -d ayuraj/asl-dataset -p data --unzip   # 2,515 images, 36 classes, 5 signers
```

## Pipeline

1. `hand_landmarks_demo.py` runs MediaPipe HandLandmarker on a webcam, video or image and writes annotated video plus a `(frames, 126)` landmark array.
2. `extract_landmarks.py` runs the landmarker over a folder-per-class image dataset once (optional border padding) and caches 21x3 landmarks per image.
3. `features.py` turns raw landmarks into wrist-centered, scale-normalized, mirror-canonical features (plus optional pairwise distances).
4. `train_baseline.py` trains and evaluates the baseline and iterations, writes `results/results.md`, confusion matrices and the fitted models.
5. `demo_predict.py` runs the trained classifier live with sliding-window majority-vote smoothing.

## Run

```bash
# baseline extraction (no padding) and padded extraction (iteration 1)
python extract_landmarks.py --data data/asl_dataset --per-class 0 --pad 0   --out cache/nopad_landmarks.npz
python extract_landmarks.py --data data/asl_dataset --per-class 0 --pad 0.5 --out cache/train_landmarks.npz
# signer-independent evaluation: train on hand1,3,4,5, test on hand2
python train_baseline.py --cache cache/nopad_landmarks.npz --out results_nopad --only baseline
python train_baseline.py --cache cache/train_landmarks.npz --out results
# live demo with the best model (normalized landmarks + logistic regression)
python demo_predict.py --model results/model_iter1.joblib
# regenerate the proposal from results/
python make_proposal.py . Final_Project_Proposal.docx
```

## Experiments (proposal, signer-independent test on hand2)

| Proposal row | Script name | Padding | Features | Model | Hand found | Test acc | Strict |
|---|---|---|---|---|---|---|---|
| Baseline | results_nopad/baseline | none | raw xyz | logistic regression | 64.5% | 82.1% | 60.6% |
| Iter 1 | results/baseline | 50% | raw xyz | logistic regression | 98.4% | 88.3% | 87.0% |
| Iter 2 | results/iter1 | 50% | normalized | logistic regression | 98.4% | 91.5% | 90.2% |
| Iter 3 | results/iter2 | 50% | normalized + pairwise distances | MLP (256,128) | 98.4% | 90.6% | 89.3% |

"Strict" counts an image with no detected hand as an error. `train_baseline.py --split block` instead holds out the last 20% of each class in file order.

## Midterm iterations (Nov 2026)

```bash
# iteration 4: landmark augmentation grid, signer-independent test on hand2
python train_augment.py --cache cache/train_landmarks.npz --out results_aug --copies 5

# iteration 5: record our own data (each person, two sessions), then transfer experiments
python record_signs.py --signer isha --session window
python record_signs.py --signer isha --session room
python record_signs.py --signer eva --session window
python record_signs.py --signer eva --session room
python train_own.py --out results_own

# live demo with the final model
python demo_predict.py --model results_own/model_final.joblib
```

| Script | What it does |
|---|---|
| `augment.py` | rotate / scale / tilt / jitter augmentations on (N, 21, 3) landmarks |
| `train_augment.py` | iteration 4 grid: each augmentation alone and combined, logreg and MLP, 3 seeds. Writes `results_aug/results.md` and `augmentation_bar.png` |
| `record_signs.py` | webcam recorder, prompts one sign at a time and saves landmarks to `data_own/<signer>_<session>.npz` |
| `train_own.py` | iteration 5: Kaggle -> us, other person -> me, Kaggle + other + aug -> me. Writes `results_own/` |

Iteration 4 result (hand2 held out, logistic regression, 3 seeds): no augmentation 91.3%, tilt+jitter 93.3 +/- 0.2%, all four 93.1%. In-plane rotation augmentation hurts (90.0%) and so does rotation-normalizing the features (90.8%), because several letters (G, H, P, Q) are told apart partly by which way the hand points.

`pixel_baseline.py` is a raw-pixel / HOG fallback for datasets where landmark extraction is impossible (e.g. images with a skeleton already drawn on them).
