# thrml-lunar

A 9×9 Lunar Magic Square solver with two interchangeable samplers:

- **Simulated annealing** (default) — pure NumPy, no extra dependencies
- **THRML** (optional) — Extropic's JAX-based probabilistic sampler; the factor-graph stub in `thrml_sampler()` is ready to wire up

Magic constant for a 9×9 normal magic square: **369**

---

## Quick start

```bash
# No extra deps required
python lunar_magic_thrml.py --mode simulate --steps 50000 --plot --save_dir ./runs
```

Outputs:
- `runs/lunar_magic_9x9.csv` — final grid
- `runs/run_summary.json` — metadata (steps, accept rate, energy, elapsed)
- `runs/lunar_magic_9x9.png` — heatmap (requires `--plot` and `matplotlib`)

---

## Options

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `simulate` | `simulate` or `thrml` |
| `--steps` | `30000` | More steps → lower energy |
| `--temp_start` | `5.0` | Annealing start temperature |
| `--temp_end` | `0.1` | Annealing end temperature |
| `--seed` | `7` | RNG seed |
| `--plot` | off | Save heatmap PNG |
| `--save_dir` | `./runs` | Output directory |
| `--diagonal` | off | Enforce main + anti-diagonal constraints |
| `--no-uniqueness` | off | Disable duplicate penalty (faster, looser) |
| `--lunar_weight` | `0.0` | Weight for lunar symbolism bias term |

---

## Lunar symbolism

`lunar_symbols()` maps rows/cols onto phase gradients (new → full → new across 9 rows, decans across 9 cols). Pass `--lunar_weight 5.0` to bias the cooled grid toward these gradients.

```bash
python lunar_magic_thrml.py --mode simulate --steps 30000 --lunar_weight 5.0 --plot
```

---

## Sigil renderer

`lunar_sigil.py` traces a path through the grid visiting cells 1 → 81 in order, producing a sigil line-drawing from the magic square's geometry.

```bash
python lunar_sigil.py   # reads runs/lunar_magic_9x9.csv
```

---

## THRML (optional)

```bash
pip install thrml jax jaxlib
python lunar_magic_thrml.py --mode thrml --steps 10000 --plot
```

The `thrml_sampler()` function contains scaffolded pseudocode for a factor graph (row/col sum factors, optional diagonals, uniqueness, lunar bias). Replace the pseudocode block with actual THRML API calls once the API stabilizes.

---

## Dependencies

- **Required**: `numpy`
- **Optional**: `matplotlib` (plots), `thrml` + `jax` (THRML mode)

---

*Made with care for Elura, by Mirai.*
