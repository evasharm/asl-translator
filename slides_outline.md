# Final talk: 6-slide outline (5 minutes)

Numbers come from the results files on main. Rewrite the wording in your own voice; the
speaker notes are suggestions, not a script.

## 1. Problem and audience
- Deaf and hard-of-hearing people sign; most hearing people can't read it. Goal: live captions from a laptop webcam, no special hardware.
- Two parts: fingerspelling (letter by letter, for names and rare words) and word signs.
- Note: one sentence on who would use it and where (class, office hours, a front desk).

## 2. Pipeline
- Webcam -> MediaPipe hand landmarks (21 points) -> normalize (wrist-centered, hand-length scale, mirror left hands) -> classifier -> caption rules -> optional LLM cleanup.
- Word signs: 32-frame clips of both hands -> GRU.
- Figure: pipeline diagram from the midterm (add the caption and cleanup boxes).

## 3. Baseline to midterm, and the webcam finding
| | Kaggle held-out signer | Our webcams |
|---|---|---|
| Raw landmarks, logistic regression (baseline) | 82.1% | - |
| Normalized + padding + augmentation | 93.3% | Eva 59.6%, Isha 49.6% (iteration 5) |
| + other person's recording | - | Eva 60.6%, Isha 54.9% |
- Key point: 91% on a held-out Kaggle signer becomes 45 to 60% on a real webcam. Testing on new people and cameras is what matters.

## 4. Word signs (iteration 7)
- Google ASL Signs dataset, 4 held-out signers.
- 20 signs: frame-average 66.6% -> GRU 76.9% -> + velocity, augmentation, arms 80.6% (top-5 95.3%).
- 50 signs: 53.8% -> 70.7%.
- Biggest gain is modelling the order of frames (+10 to 13 points). Arms help on 20 signs (+2.7) but the demo only tracks hands, so it runs the 77.9% hands-only model.
- Hardest: LISTEN vs HEAR, AWAKE vs WAKE (near-identical signs).
- Figure: results_signs_20/signs_bar.png

## 5. From letters to captions (iterations 8, 9, 10)
- Character error rate on 10 fingerspelled words each, tuned on one person, tested on the other:
  every frame 539% -> majority vote 123% -> confidence-weighted 90% -> + hold rule 45%.
- Example: FIVE -> `FIVE`, CAT -> `CAT`, HELLO -> `HELO` (missed double letter).
- Two-stage landmarks on Eva's new recording: no overall gain (71.8% vs 73.9%); small help only on the smallest hands.
- LLM cleanup: [Eva: raw vs dictionary vs LLM error rate, over-correction, latency, which model goes in the demo].

## 6. Demo video, then what we'd do next
- Demo (1 to 2 min): spell two words, sign three or four trained words, raw caption turning into cleaned caption.
- Next: a sequence model trained on fingerspelling video (CTC) instead of frame classifier + rules; more signers and cameras; one stream mixing words and fingerspelling.
