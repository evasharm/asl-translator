## Iteration 9: single pass vs two-stage on eva_close (same frames)

Letter accuracy (J/Z excluded, 24 letters). Hand height = fraction of the frame height. 2 seeds.

| Train | Landmarks | Test acc | hand < 0.30 | hand 0.30-0.40 | hand 0.40-0.50 | hand >= 0.50 |
|---|---|---|---|---|---|---|
| unseen signer | single pass | 73.9 +/- 0.4% | 18.2% | 71.5% | 81.8% | 80.3% |
| new session | single pass | 81.8 +/- 0.1% | 40.7% | 69.0% | 84.4% | 89.7% |
| unseen signer | two-stage | 71.8 +/- 0.1% | 16.7% | 78.0% | 84.2% | 75.0% |
| new session | two-stage | 79.0 +/- 0.2% | 42.1% | 71.8% | 84.7% | 84.2% |

Frames per hand-height bin: < 0.30: 563, 0.30-0.40: 666, 0.40-0.50: 1277, >= 0.50: 3543

unseen signer = Kaggle + Isha's recording (no Eva data). new session = also Eva's older room recording.

Most confused letter pairs (unseen signer, two-stage; used by eval_cleanup.py): G/P (209), R/U (118), M/S (110), K/V (81), M/T (69), O/Q (66), G/O (65), G/X (57), U/V (55), D/X (53)

Final demo model: results_final/model_final.joblib, trained on Kaggle + all of data_own (28,794 samples after augmentation), two-stage landmarks.
