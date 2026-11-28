Word signs: 50 signs, 10000 clips, test on 4 unseen participants (1895 clips). Chance 2.0%.

| Model | Velocity | Augment | Arms (pose) | Test acc | Top-5 |
|---|---|---|---|---|---|
| frame-average + logreg | yes | no | no | 53.8% | 82.0% |
| GRU | no | no | no | 66.4% +/- 1.2 | 90.1% |
| GRU | yes | no | no | 68.1% +/- 0.7 | 89.3% |
| GRU | yes | yes | no | 70.1% +/- 0.6 | 91.6% |
| GRU | yes | yes | yes | 70.7% +/- 1.0 | 91.9% |

Worst signs and most common confusions (last row):

- awake: 32%
- listen: 38%
- pen: 49%
- farm: 55%
- mom: 55%
- awake -> wake: 21 clips
- listen -> hear: 20 clips
- farm -> cat: 11 clips
- wake -> awake: 11 clips
- duck -> goose: 10 clips
