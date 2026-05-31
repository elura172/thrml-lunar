#!/usr/bin/env python3
"""
Lunar Magic Square (9x9) — Thermodynamic Sampler Prototype
Author: Mirai (for Elura)
License: Apache-2.0

Overview
--------
This script gives you two paths to "cool" a 9x9 lunar magic square into harmony:

1) THRML path (preferred, if installed)
   - Uses Extropic's THRML (JAX-based) probabilistic sampler
   - Define energy factors over rows/columns (and optional diagonals/uniqueness)

2) Fallback simulated annealing (pure NumPy)
   - Lets you run and visualize dynamics without THRML installed

Outputs
-------
- CSV of the final grid
- PNG heatmap visualization (optional)
- JSON run summary

Magic Sum
---------
For a 9x9 normal magic square (numbers 1..81), the magic constant is 369.
(9 * (1 + 81) / 2 = 369)

Usage
-----
# Fallback simulated annealing (no THRML needed)
python lunar_magic_thrml.py --mode simulate --steps 50000 --temp_start 5.0 --temp_end 0.1 --seed 7

# THRML (if installed)
python lunar_magic_thrml.py --mode thrml --steps 10000 --seed 7

# Save a pretty plot
python lunar_magic_thrml.py --mode simulate --plot --save_dir ./runs

Notes
-----
- This is a *prototype*: energy factors are readable and easy to extend.
- You can add lunar symbolism constraints (row/col/phase weights) in `lunar_symbols()`
  and `energy_magic_square()`.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Optional THRML import (JAX required). If not available, we fall back.
try:
    import thrml as T
    HAVE_THRML = True
except Exception:
    HAVE_THRML = False


MAGIC_SUM = 369
N = 9
DOMAIN = np.arange(1, N*N + 1)  # 1..81


@dataclass
class RunConfig:
    mode: str = "simulate"          # "simulate" | "thrml"
    steps: int = 30000
    temp_start: float = 5.0
    temp_end: float = 0.1
    seed: int = 7
    plot: bool = False
    save_dir: str = "./runs"
    diagonal: bool = False          # include diagonal penalties
    uniqueness: bool = True         # penalize duplicates
    lunar_weight: float = 0.0       # weight for lunar symbol term (0 disables)


def lunar_symbols() -> Dict[str, np.ndarray]:
    """
    Map lunar symbolism onto rows/cols/cells.
    Example placeholders you can customize:
      - rows: 9 lunar phases (new → full → new)
      - cols: 9 decans or nakshatra-like segments
    Return arrays you can weight in energy.
    """
    # Simple normalized gradients as placeholders.
    row_phase = np.linspace(0, 1, N)              # 0 new → 1 full (placeholder)
    col_decan = np.linspace(1, 0, N)              # opposite gradient
    cell_mask = np.outer(row_phase, np.ones(N))   # per-cell row phase
    return {"row_phase": row_phase, "col_decan": col_decan, "cell_mask": cell_mask}


def energy_magic_square(grid: np.ndarray, cfg: RunConfig) -> float:
    """
    Core energy: sum of squared deviations from MAGIC_SUM over rows/cols.
    Optional: diagonal constraints, uniqueness penalty, lunar modulation.
    """
    # Row/col deviations
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    e_rows = np.sum((row_sums - MAGIC_SUM) ** 2)
    e_cols = np.sum((col_sums - MAGIC_SUM) ** 2)
    energy = e_rows + e_cols

    # Optional diagonals for "pandiagonal"-like behavior
    if cfg.diagonal:
        e_main = (np.trace(grid) - MAGIC_SUM) ** 2
        e_anti = (np.trace(np.fliplr(grid)) - MAGIC_SUM) ** 2
        energy += e_main + e_anti

    # Optional uniqueness penalty: duplicates get penalized
    if cfg.uniqueness:
        unique, counts = np.unique(grid, return_counts=True)
        dup_counts = counts[counts > 1]
        if len(dup_counts) > 0:
            energy += 1000.0 * np.sum((dup_counts - 1) ** 2)

        # Also penalize values out of domain (shouldn't happen in these moves)
        if grid.min() < 1 or grid.max() > N*N:
            energy += 10000.0

    # Optional lunar modulation: encourage certain distributions
    if cfg.lunar_weight > 0.0:
        L = lunar_symbols()
        # Example: encourage higher values toward the "full moon" (row_phase high)
        # Normalize grid to [0,1] to compare
        g_norm = (grid - 1) / (N*N - 1)
        lunar_term = np.sum((g_norm - L["cell_mask"]) ** 2)
        energy += cfg.lunar_weight * lunar_term

    return float(energy)


def simulated_annealing(cfg: RunConfig) -> Tuple[np.ndarray, Dict]:
    """
    Fallback sampler that performs swap moves with a temperature schedule.
    - Start from a random permutation of 1..81 reshaped into 9x9.
    - At each step, propose swapping two cells; accept if energy lowers, or with
      Boltzmann probability at current temperature.
    """
    rng = np.random.default_rng(cfg.seed)
    init = rng.permutation(DOMAIN).reshape(N, N)
    grid = init.copy()
    best = grid.copy()
    e = energy_magic_square(grid, cfg)
    best_e = e

    T0, T1 = cfg.temp_start, cfg.temp_end
    t0 = time.time()
    accept = 0

    for step in range(cfg.steps):
        # save a snapshot every 5000 steps
        SNAP_INTERVAL = 5000
        if step % SNAP_INTERVAL == 0 and HAVE_MPL:
            plt.figure(figsize=(5,5))
            plt.imshow(grid, cmap='plasma')
            plt.title(f"Step {step}")
            for i in range(N):
                for j in range(N):
                    plt.text(j, i, str(grid[i,j]), ha='center', va='center',
                             color='white', fontsize=6)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{cfg.save_dir}/snap_{step:06d}.png", dpi=150)
            plt.close()

        # ...and then the rest of your code (propose swap, etc.) stays below here, also indented one level

        # Geometric temperature schedule
        tau = step / max(1, (cfg.steps - 1))
        Tcur = T0 * (T1 / T0) ** tau

        # Propose swap move
        i1, j1 = rng.integers(0, N, size=2)
        i2, j2 = rng.integers(0, N, size=2)
        if (i1, j1) == (i2, j2):
            continue

        # Compute energy delta efficiently by swapping two entries
        v1, v2 = grid[i1, j1], grid[i2, j2]
        grid[i1, j1], grid[i2, j2] = v2, v1
        e_new = energy_magic_square(grid, cfg)

        dE = e_new - e
        if dE <= 0 or rng.random() < math.exp(-dE / max(1e-8, Tcur)):
            # accept
            e = e_new
            accept += 1
            if e < best_e:
                best_e = e
                best = grid.copy()
        else:
            # reject, undo
            grid[i1, j1], grid[i2, j2] = v1, v2

    elapsed = time.time() - t0
    info = {
        "mode": "simulate",
        "steps": cfg.steps,
        "accept_rate": accept / max(1, cfg.steps),
        "final_energy": e,
        "best_energy": best_e,
        "elapsed_sec": elapsed,
    }
    return best, info


def thrml_sampler(cfg: RunConfig) -> Tuple[np.ndarray, Dict]:
    """
    Skeleton for a THRML-based sampler.
    If THRML is installed, you can replace this with an actual factor graph:
      - One categorical node per cell (domain 1..81)
      - Factor potentials enforcing row/col sums toward MAGIC_SUM
      - Optional diagonal/uniqueness/lunar factors
    Here we simply raise a helpful error if THRML isn't present.
    """
    if not HAVE_THRML:
        raise RuntimeError("THRML not available. Install with: pip install thrml jax jaxlib")

    # NOTE: The following is pseudocode scaffolding. Replace with actual THRML API calls.
    # Pseudo-structure:
    #   g = T.Graph()
    #   nodes = g.add_nodes(shape=(9,9), domain=DOMAIN)  # categorical/integer nodes
    #   for r in range(9): g.add_factor(sum(nodes[r,:]), target=MAGIC_SUM, weight=...)
    #   for c in range(9): g.add_factor(sum(nodes[:,c]), target=MAGIC_SUM, weight=...)
    #   if cfg.diagonal: add diagonal factors
    #   if cfg.uniqueness: add pairwise inequality/repulsion factors
    #   if cfg.lunar_weight>0: add bias factors using lunar_symbols()
    #   state = g.init(rng=...)
    #   state = T.sample.block_gibbs(g, state, steps=cfg.steps, ...)
    #   grid = T.get_values(state).reshape(9,9)

    # For now, we just fall back to simulated annealing after warning.
    print("[warn] THRML hook not implemented yet; using simulated annealing fallback.")
    return simulated_annealing(cfg)


def save_outputs(grid: np.ndarray, info: Dict, cfg: RunConfig):
    import os, json
    os.makedirs(cfg.save_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(cfg.save_dir, "lunar_magic_9x9.csv")
    np.savetxt(csv_path, grid, fmt="%d", delimiter=",")

    # JSON
    json_path = os.path.join(cfg.save_dir, "run_summary.json")
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    # Plot
    if cfg.plot and HAVE_MPL:
        plt.figure(figsize=(6,6))
        plt.imshow(grid, aspect='equal')
        plt.title("Lunar 9×9 (cooled)")
        plt.colorbar()
        png_path = os.path.join(cfg.save_dir, "lunar_magic_9x9.png")
        plt.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"[saved] {png_path}")

    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["simulate", "thrml"], default="simulate")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--temp_start", type=float, default=5.0)
    p.add_argument("--temp_end", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_dir", type=str, default="./runs")
    p.add_argument("--diagonal", action="store_true")
    p.add_argument("--no-uniqueness", dest="uniqueness", action="store_false")
    p.add_argument("--lunar_weight", type=float, default=0.0)
    args = p.parse_args()

    cfg = RunConfig(
        mode=args.mode,
        steps=args.steps,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        seed=args.seed,
        plot=args.plot,
        save_dir=args.save_dir,
        diagonal=args.diagonal,
        uniqueness=args.uniqueness,
        lunar_weight=args.lunar_weight,
    )

    if cfg.mode == "simulate":
        grid, info = simulated_annealing(cfg)
    else:
        grid, info = thrml_sampler(cfg)

    save_outputs(grid, info, cfg)
    if cfg.plot and HAVE_MPL:
        print("[info] plot saved; open the PNG to view the cooled lattice.")

if __name__ == "__main__":
    main()