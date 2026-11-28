## Iteration 9: single pass vs two-stage on eva_close (same frames)

Letter accuracy (J/Z excluded, 24 letters). Hand size = longer side of the hand's box as a fraction of the frame height. 2 seeds.

| Train | Landmarks | Test acc | size < 0.45 | size 0.45-0.55 | size 0.55-0.65 | size >= 0.65 |
|---|---|---|---|---|---|---|
| unseen signer | single pass | 73.9 +/- 0.4% | 62.3% | 70.5% | 83.2% | 74.3% |
| new session | single pass | 81.8 +/- 0.1% | 75.4% | 72.5% | 89.4% | 84.7% |
| unseen signer | two-stage | 71.8 +/- 0.1% | 65.6% | 67.8% | 78.6% | 71.6% |
| new session | two-stage | 79.0 +/- 0.2% | 77.2% | 69.4% | 83.4% | 82.5% |

Frames per hand-size bin: < 0.45: 1320, 0.45-0.55: 1216, 0.55-0.65: 2003, >= 0.65: 1510

unseen signer = Kaggle + Isha's recording (no Eva data). new session = also Eva's older room recording.

Most confused letter pairs (unseen signer, two-stage; used by eval_cleanup.py): G/P (209), R/U (118), M/S (110), K/V (81), M/T (69), O/Q (66), G/O (65), G/X (57), U/V (55), D/X (53)

Final demo model: results_final/model_final.joblib, trained on Kaggle + all of data_own (28,794 samples after augmentation), two-stage landmarks.
