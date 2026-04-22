"""Sample n points uniformly from SVG path segments (each segment equally
likely) and plot them next to the original SVG."""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svgstr2paths

from classifier_svg import load_one_svg, render_svg_to_image


def sample_points(svg_xml, n, rng):
    paths, _ = svgstr2paths(svg_xml)
    segs = [s for p in paths for s in p]
    m = re.search(r'viewBox="([^"]+)"', svg_xml)
    x0, y0, w, h = map(float, m.group(1).split()) if m else (0.0, 0.0, 1.0, 1.0)
    idx = rng.integers(0, len(segs), size=n)
    t = rng.random(n)
    pts = np.array([[(s := segs[i]).point(tt).real, s.point(tt).imag]
                    for i, tt in zip(idx, t)])
    pts[:, 0] = (pts[:, 0] - x0) / w
    pts[:, 1] = (pts[:, 1] - y0) / h
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--skip", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="distribute_n_points_on_boundaries.png")
    args = ap.parse_args()

    svg_xml, desc = load_one_svg(skip=args.skip)
    pts = sample_points(svg_xml, args.n, np.random.default_rng(args.seed))
    img = render_svg_to_image(svg_xml, 512)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, extent=[0, 1, 1, 0])
    axes[0].set_title(f"Original SVG\n{desc[:60]}", fontsize=9)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=2, c="black")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(1, 0)
    axes[1].set_title(f"{args.n} points on segments (uniform per segment)", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
