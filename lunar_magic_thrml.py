#!/usr/bin/env python3
"""
Lunar Magic Square (9x9) — Simulated Annealing Solver
Author: Mirai (for Elura)
License: Apache-2.0

Cools a 9x9 magic square into harmony using swap-based simulated annealing.
Each move swaps two cells, preserving the permutation structure (all values
1..81 used exactly once). This is fundamentally why SA works here and why
Gibbs-style samplers (THRML, block Gibbs) do not: Gibbs resamples one cell
in isolation, immediately violating uniqueness; SA trades two cells atomically.

Magic constant: 9 * (1 + 81) / 2 = 369

Usage
-----
python lunar_magic_thrml.py --steps 50000 --plot --save_dir ./runs
python lunar_magic_thrml.py --steps 500000 --diagonal   # harder: enforce diagonals too
python lunar_magic_thrml.py --steps 30000 --lunar_weight 5.0 --plot
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


MAGIC_SUM = 369
N = 9
DOMAIN = np.arange(1, N * N + 1)  # 1..81


@dataclass
class RunConfig:
    steps: int = 30000
    temp_start: float = 5.0
    temp_end: float = 0.1
    seed: int = 7
    plot: bool = False
    save_dir: str = "./runs"
    diagonal: bool = False
    lunar_weight: float = 0.0


def lunar_symbols() -> Dict[str, np.ndarray]:
    """
    Lunar symbolism hooks — map rows/cols onto phase gradients.
      rows: 9 phases, new moon (0) → full moon (1) → new moon
      cols: 9 decans, descending
    Edit these to taste; used when --lunar_weight > 0.
    """
    row_phase = np.linspace(0, 1, N)
    col_decan = np.linspace(1, 0, N)
    cell_mask = np.outer(row_phase, np.ones(N))
    return {"row_phase": row_phase, "col_decan": col_decan, "cell_mask": cell_mask}


def energy(grid: np.ndarray, cfg: RunConfig) -> float:
    """
    Energy = sum of squared deviations from MAGIC_SUM across all rows and cols.
    Lower is better; 0 is a perfect (semi-)magic square.
    Optional: diagonal constraints, lunar phase bias.
    """
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    e = float(np.sum((row_sums - MAGIC_SUM) ** 2) + np.sum((col_sums - MAGIC_SUM) ** 2))

    if cfg.diagonal:
        e += (np.trace(grid) - MAGIC_SUM) ** 2
        e += (np.trace(np.fliplr(grid)) - MAGIC_SUM) ** 2

    if cfg.lunar_weight > 0.0:
        L = lunar_symbols()
        g_norm = (grid - 1) / (N * N - 1)
        e += cfg.lunar_weight * float(np.sum((g_norm - L["cell_mask"]) ** 2))

    return e


def simulated_annealing(cfg: RunConfig) -> Tuple[np.ndarray, Dict]:
    """
    Swap-based simulated annealing over permutations of 1..81.

    Move: pick two distinct cells, swap their values. Uniqueness is always
    preserved — no penalty term needed. Accept if energy drops, otherwise
    accept with Boltzmann probability exp(-dE / T).
    """
    os.makedirs(cfg.save_dir, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    grid = rng.permutation(DOMAIN).reshape(N, N)
    best = grid.copy()
    e = energy(grid, cfg)
    best_e = e

    T0, T1 = cfg.temp_start, cfg.temp_end
    t0 = time.time()
    accept = 0

    for step in range(cfg.steps):
        if step % 5000 == 0 and HAVE_MPL:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(grid, cmap="plasma")
            ax.set_title(f"Step {step}  E={e:.0f}")
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                            color="white", fontsize=6)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(f"{cfg.save_dir}/snap_{step:06d}.png", dpi=150)
            plt.close(fig)

        tau = step / max(1, cfg.steps - 1)
        Tcur = T0 * (T1 / T0) ** tau

        i1, j1 = rng.integers(0, N, size=2)
        i2, j2 = rng.integers(0, N, size=2)
        if (i1, j1) == (i2, j2):
            continue

        v1, v2 = grid[i1, j1], grid[i2, j2]
        grid[i1, j1], grid[i2, j2] = v2, v1
        e_new = energy(grid, cfg)

        dE = e_new - e
        if dE <= 0 or rng.random() < math.exp(-dE / max(1e-8, Tcur)):
            e = e_new
            accept += 1
            if e < best_e:
                best_e = e
                best = grid.copy()
        else:
            grid[i1, j1], grid[i2, j2] = v1, v2

    elapsed = time.time() - t0
    return best, {
        "steps": cfg.steps,
        "accept_rate": round(accept / max(1, cfg.steps), 4),
        "final_energy": e,
        "best_energy": best_e,
        "elapsed_sec": round(elapsed, 3),
    }


def save_outputs(grid: np.ndarray, info: Dict, cfg: RunConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    csv_path = os.path.join(cfg.save_dir, "lunar_magic_9x9.csv")
    np.savetxt(csv_path, grid, fmt="%d", delimiter=",")

    json_path = os.path.join(cfg.save_dir, "run_summary.json")
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    if cfg.plot and HAVE_MPL:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid, aspect="equal")
        ax.set_title("Lunar 9×9 (cooled)")
        fig.colorbar(im)
        png_path = os.path.join(cfg.save_dir, "lunar_magic_9x9.png")
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {png_path}")

    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")
    print(f"[info]  best_energy={info['best_energy']}  steps={info['steps']}  "
          f"accept_rate={info['accept_rate']}  elapsed={info['elapsed_sec']}s")


def main():
    p = argparse.ArgumentParser(description="9x9 Lunar Magic Square — simulated annealing")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--temp_start", type=float, default=5.0)
    p.add_argument("--temp_end", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_dir", type=str, default="./runs")
    p.add_argument("--diagonal", action="store_true", help="also enforce diagonal = 369")
    p.add_argument("--lunar_weight", type=float, default=0.0,
                   help="bias toward lunar phase gradient (0 = off)")
    args = p.parse_args()

    cfg = RunConfig(
        steps=args.steps,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        seed=args.seed,
        plot=args.plot,
        save_dir=args.save_dir,
        diagonal=args.diagonal,
        lunar_weight=args.lunar_weight,
    )

    grid, info = simulated_annealing(cfg)
    save_outputs(grid, info, cfg)


if __name__ == "__main__":
    main()
