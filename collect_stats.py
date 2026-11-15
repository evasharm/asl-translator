"""Gather numbers for the proposal into results/: extraction stats, demo-clip
stats (parsed from results_demo.log) and a side-by-side confusion figure."""
import json
import os
import re
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
R = os.path.join(ROOT, "results")
os.makedirs(R, exist_ok=True)

shutil.copy(os.path.join(ROOT, "cache", "train_landmarks.stats.json"), os.path.join(R, "extraction_stats.json"))

demo = dict(n_frames=0, det_rate=0.0, model_fps=0.0, e2e_fps=0.0)
log = os.path.join(ROOT, "results_demo.log")
if os.path.exists(log):
    txt = open(log).read()
    g = lambda pat: re.search(pat, txt)
    if g(r"Frames processed:\s+(\d+)"):
        demo["n_frames"] = int(g(r"Frames processed:\s+(\d+)").group(1))
        demo["det_rate"] = float(g(r"Frames with a hand:\s+\d+ \(([\d.]+)%\)").group(1)) / 100
        demo["model_fps"] = float(g(r"Model speed \(smoothed\):\s+([\d.]+)").group(1))
        demo["e2e_fps"] = float(g(r"End to end throughput:\s+([\d.]+)").group(1))
json.dump(demo, open(os.path.join(R, "demo_stats.json"), "w"), indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

a, b = (os.path.join(R, f"confusion_{n}.png") for n in ("baseline", "iter2"))
if os.path.exists(a) and os.path.exists(b):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, p in zip(axes, (a, b)):
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(R, "confusion_pair.png"), dpi=170)
print(json.dumps(demo))
