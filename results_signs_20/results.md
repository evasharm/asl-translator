Word signs: 20 signs, 6000 clips, test on 4 unseen participants (1202 clips). Chance 5.0%.

| Model | Velocity | Augment | Arms (pose) | Test acc | Top-5 |
|---|---|---|---|---|---|
| frame-average + logreg | yes | no | no | 66.6% | 90.8% |
| GRU | no | no | no | 76.9% +/- 0.3 | 94.4% |
| GRU | yes | no | no | 76.4% +/- 1.6 | 94.5% |
| GRU | yes | yes | no | 77.9% +/- 0.3 | 95.5% |
| GRU | yes | yes | yes | 80.6% +/- 0.1 | 95.3% |

Worst signs and most common confusions (last row):

- listen: 35%
- bye: 65%
- nuts: 67%
- pretend: 68%
- hear: 74%
- listen -> hear: 30 clips
- shhh -> mouse: 11 clips
- hear -> listen: 10 clips
- pretend -> mouse: 10 clips
- bird -> mouse: 7 clips
