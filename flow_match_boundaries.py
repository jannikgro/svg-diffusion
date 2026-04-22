"""Flow-matching model that learns the distribution of points lying on the
visible path segments of a single SVG.

Training points are drawn with the same 'pick a random segment, pick a random
t' scheme as distribute_n_points_on_boundaries.py. We pre-sample a large pool
once (svgpathtools segment.point is a Python call, so doing it fresh per step
would dominate runtime); each training step draws a random slice of size
batch_size from that pool.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from classifier_svg import load_one_svg, render_svg_to_image
from distribute_n_points_on_boundaries import sample_points

PLOTS_DIR = "boundary_flow_plots"


class FlowMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, xy, t):
        return self.net(torch.cat([xy, t[:, None]], dim=-1))


@torch.no_grad()
def euler_sample(model, n, steps, device):
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x


def make_fig(svg_img, pts, desc, step):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(svg_img, extent=[0, 1, 1, 0])
    axes[0].set_title(f"Original SVG\n{desc[:60]}", fontsize=9)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=2, c="black")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(1, 0)
    axes[1].set_title(f"Flow samples (step {step})", fontsize=9)
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--euler-steps", type=int, default=128)
    ap.add_argument("--n-plot-points", type=int, default=4096)
    ap.add_argument("--pool-size", type=int, default=1_000_000)
    ap.add_argument("--skip", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-log-every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    svg_xml, desc = load_one_svg(skip=args.skip)
    print(f"Loaded SVG: {desc[:80]}")

    rng = np.random.default_rng(args.seed)
    print(f"Pre-sampling {args.pool_size} training points on segments...")
    pool = sample_points(svg_xml, args.pool_size, rng)
    # Data -> [-1, 1] so it lives on a similar scale to the N(0,I) source.
    pool = torch.from_numpy(pool.astype(np.float32) * 2.0 - 1.0).to(device)

    svg_img = render_svg_to_image(svg_xml, 512)

    torch.manual_seed(args.seed)
    model = FlowMLP(hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    wandb.init(project="svg_boundary_flow", config=vars(args))

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, args.pool_size, (args.batch_size,), device=device)
        x1 = pool[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(args.batch_size, device=device)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        loss = F.mse_loss(model(xt, t), x1 - x0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.loss_log_every == 0:
            wandb.log({"loss": loss.item()}, step=step)
        if step % args.sample_every == 0:
            model.eval()
            samples = euler_sample(model, args.n_plot_points, args.euler_steps, device)
            model.train()
            pts = (samples.cpu().numpy() + 1.0) / 2.0
            fig = make_fig(svg_img, pts, desc, step)
            out_path = os.path.join(PLOTS_DIR, f"samples_{step:08d}.png")
            fig.savefig(out_path, dpi=150)
            wandb.log({"samples": wandb.Image(fig)}, step=step)
            plt.close(fig)
            print(f"step {step:7d}  loss {loss.item():.4f}  saved {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
