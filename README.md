# thrml-lunar

A 9×9 Lunar Magic Square solver using swap-based simulated annealing.

Magic constant: **369** (= 9 × (1 + 81) / 2)

---

## Why simulated annealing?

Magic squares are a **permutation problem** — all 81 values must appear exactly once. SA handles this naturally: each move swaps two cells, so uniqueness is always preserved and no penalty term is needed.

Gibbs-style samplers (THRML, block Gibbs) resample one cell at a time from its conditional distribution. That move immediately violates uniqueness — 80 of 81 possible values are already taken, leaving the sampler frozen in place. The THRML path was explored and found to be architecturally mismatched to permutation constraints; SA is the correct tool here.

---

## Quick start

```bash
python lunar_magic_thrml.py --steps 50000 --plot --save_dir ./runs
```

Outputs:
- `runs/lunar_magic_9x9.csv` — final grid
- `runs/run_summary.json` — steps, accept rate, energy, elapsed time
- `runs/lunar_magic_9x9.png` — heatmap (requires `--plot` and `matplotlib`)

---

## Options

| Flag | Default | Notes |
|---|---|---|
| `--steps` | `30000` | More steps → lower energy. 50k reliably hits energy 0. |
| `--temp_start` | `5.0` | Annealing start temperature |
| `--temp_end` | `0.1` | Annealing end temperature |
| `--seed` | `7` | RNG seed |
| `--plot` | off | Save heatmap PNG + per-5000-step snapshots |
| `--save_dir` | `./runs` | Output directory |
| `--diagonal` | off | Also enforce main + anti-diagonal = 369 (much harder; use 500k+ steps) |
| `--lunar_weight` | `0.0` | Bias grid toward lunar phase gradient (0 = off) |

---

## Energy

Energy = Σ (row\_sum − 369)² + Σ (col\_sum − 369)²

- **0** = perfect semi-magic square (all rows + cols hit 369)
- With `--diagonal`: also penalizes main and anti-diagonal deviations

---

## Lunar symbolism

`lunar_symbols()` maps rows onto a new→full→new phase gradient and cols onto a decan gradient. Pass `--lunar_weight 5.0` to bias the cooled grid toward these gradients.

```bash
python lunar_magic_thrml.py --steps 30000 --lunar_weight 5.0 --plot
```

---

## Sigil renderer

`lunar_sigil.py` traces a path through the solved grid visiting cells 1 → 81 in order, producing a line-drawing sigil from the magic square's geometry.

```bash
python lunar_sigil.py   # reads runs/lunar_magic_9x9.csv
```

---

## Dependencies

- **Required**: `numpy`
- **Optional**: `matplotlib` (plots and snapshots)

---

*Made with care for Elura, by Mirai.*
