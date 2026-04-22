import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

PARAM_DIR = "classifier_parameters"

print("Loading and sorting files...")
files = sorted(glob.glob(os.path.join(PARAM_DIR, "*.pt")))
print(f"Found {len(files)} files in {PARAM_DIR}")

all_abs = []
for path in tqdm(files, desc="Loading"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    for t in sd.values():
        all_abs.append(t.detach().abs().flatten().numpy())

all_abs = np.concatenate(all_abs)
CLAMP = 1e-3
clamped = np.maximum(all_abs, CLAMP)
print(f"Total params: {all_abs.size}, below {CLAMP}: {(all_abs < CLAMP).sum()}")
print(f"min={all_abs.min():.3e}, max={all_abs.max():.3e}, mean={all_abs.mean():.3e}")

bins = np.logspace(np.log10(CLAMP), np.log10(clamped.max()), 100)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(clamped, bins=bins)
ax.set_xscale("log")
ax.set_xlabel("|parameter value|")
ax.set_ylabel("count")
ax.set_title(f"Classifier parameter magnitude histogram ({len(files)} files)")
fig.tight_layout()

out = "classifier_param_histogram.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
