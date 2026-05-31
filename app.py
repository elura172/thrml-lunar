"""
Lunar Magic Square — Live Cooling UI
Streamlit app: watch the grid cool in real time, sigil path updating each step.

Two modes:
  Simulated Annealing  — swap-based, permutation-preserving, finds true magic squares
  THRML (relaxed)      — block Gibbs via JAX, uniqueness dropped so Gibbs can move freely;
                         finds grids where every row/col sums to 369 (values may repeat)

Run:
  streamlit run app.py
"""

import math
import itertools
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import numpy as np
import streamlit as st

N = 9
MAGIC_SUM = 369
DOMAIN = np.arange(1, N * N + 1)


# ── Energy ────────────────────────────────────────────────────────────────────

def compute_energy(grid: np.ndarray, diagonal: bool = False, lunar_weight: float = 0.0) -> float:
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    e = float(np.sum((row_sums - MAGIC_SUM) ** 2) + np.sum((col_sums - MAGIC_SUM) ** 2))
    if diagonal:
        e += float((np.trace(grid) - MAGIC_SUM) ** 2 + (np.trace(np.fliplr(grid)) - MAGIC_SUM) ** 2)
    if lunar_weight > 0.0:
        row_phase = np.linspace(0, 1, N)
        cell_mask = np.outer(row_phase, np.ones(N))
        g_norm = (grid - 1) / (N * N - 1)
        e += lunar_weight * float(np.sum((g_norm - cell_mask) ** 2))
    return e


# ── Generators ────────────────────────────────────────────────────────────────

def sa_generator(steps, seed, temp_start, temp_end, diagonal, lunar_weight, yield_every=300):
    rng = np.random.default_rng(seed)
    grid = rng.permutation(DOMAIN).reshape(N, N)
    e = compute_energy(grid, diagonal, lunar_weight)
    best, best_e = grid.copy(), e
    accept = 0

    for step in range(steps):
        tau = step / max(1, steps - 1)
        Tcur = temp_start * (temp_end / temp_start) ** tau

        i1, j1 = rng.integers(0, N, size=2)
        i2, j2 = rng.integers(0, N, size=2)
        if (i1, j1) == (i2, j2):
            continue

        v1, v2 = grid[i1, j1], grid[i2, j2]
        grid[i1, j1], grid[i2, j2] = v2, v1
        e_new = compute_energy(grid, diagonal, lunar_weight)
        dE = e_new - e

        if dE <= 0 or rng.random() < math.exp(-dE / max(1e-8, Tcur)):
            e = e_new
            accept += 1
            if e < best_e:
                best_e, best = e, grid.copy()
        else:
            grid[i1, j1], grid[i2, j2] = v1, v2

        if step % yield_every == 0:
            yield grid.copy(), e, best_e, step, round(accept / max(1, step + 1), 4), Tcur

    yield best, best_e, best_e, steps, round(accept / max(1, steps), 4), temp_end


def thrml_relaxed_generator(steps, seed, chunk_size=150):
    """
    THRML Option A — relaxed uniqueness.
    Values can repeat; row/col sum factors drive the grid toward 369.
    Block Gibbs can move freely because each cell's value is independent.
    """
    import jax
    import jax.numpy as jnp
    from thrml import CategoricalNode, Block, BlockGibbsSpec, SamplingSchedule, sample_states
    from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional
    from thrml.factor import FactorSamplingProgram

    K = N * N
    vals = np.arange(1, K + 1, dtype=np.float32)
    key = jax.random.PRNGKey(seed)

    # Nodes
    grid_nodes = [[CategoricalNode() for _ in range(N)] for _ in range(N)]
    flat_nodes = [grid_nodes[i][j] for i in range(N) for j in range(N)]

    # Factors (no uniqueness)
    factors = []

    # Unary: W[cell, v] = 2*(2·M·v - v²)
    w_unary = np.tile(2.0 * (2.0 * MAGIC_SUM * vals - vals ** 2), (K, 1))
    factors.append(CategoricalEBMFactor(
        node_groups=[Block(flat_nodes)],
        weights=jnp.array(w_unary),
    ))

    # Pairwise rows + cols: W[pair, vi, vj] = -2·vi·vj
    w_pair_base = -2.0 * np.outer(vals, vals)
    row_pairs = list(itertools.combinations(range(N), 2))

    for group in (
        [(grid_nodes[r][i], grid_nodes[r][j]) for r in range(N) for (i, j) in row_pairs],
        [(grid_nodes[i][c], grid_nodes[j][c]) for c in range(N) for (i, j) in row_pairs],
    ):
        heads = [p[0] for p in group]
        tails = [p[1] for p in group]
        w = np.stack([w_pair_base] * len(heads))
        factors.append(CategoricalEBMFactor(
            node_groups=[Block(heads), Block(tails)],
            weights=jnp.array(w),
        ))

    free_blocks = [Block([n]) for n in flat_nodes]
    gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks=[])
    samplers = [CategoricalGibbsConditional(n_categories=K) for _ in free_blocks]
    program = FactorSamplingProgram(gibbs_spec, samplers, factors, [])

    # Random init (not a permutation — values can repeat)
    rng = np.random.default_rng(seed)
    init_indices = rng.integers(0, K, size=K).astype(np.uint8)
    current_state = [jnp.array([v]) for v in init_indices]

    schedule = SamplingSchedule(n_warmup=chunk_size, n_samples=1, steps_per_sample=1)

    n_chunks = max(1, steps // chunk_size)
    for chunk in range(n_chunks):
        key, subkey = jax.random.split(key)
        results = sample_states(subkey, program, schedule, current_state, [], free_blocks)
        current_state = [r[-1] for r in results]

        indices = np.array([int(r[-1, 0]) for r in results], dtype=int)
        grid = (indices + 1).reshape(N, N)
        e = compute_energy(grid)

        yield grid, e, e, (chunk + 1) * chunk_size, 0.0, 0.0


# ── Renderers ─────────────────────────────────────────────────────────────────

BG = "#0e1117"
ACCENT = "#a78bfa"


def render_grid(grid: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.imshow(grid, cmap="plasma", vmin=1, vmax=81, aspect="equal")

    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)

    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(grid[i, j]),
                    ha="center", va="center", color="white",
                    fontsize=7.5, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    for i, rs in enumerate(row_sums):
        color = "#4ade80" if rs == MAGIC_SUM else "#f87171"
        ax.text(N - 0.5 + 0.7, i, f"{rs}", ha="left", va="center",
                color=color, fontsize=7, fontweight="bold")

    for j, cs in enumerate(col_sums):
        color = "#4ade80" if cs == MAGIC_SUM else "#f87171"
        ax.text(j, N - 0.5 + 0.6, f"{cs}", ha="center", va="top",
                color=color, fontsize=7, fontweight="bold")

    ax.set_xlim(-0.5, N + 0.8)
    ax.set_ylim(N + 0.9, -0.5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def render_sigil(grid: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Map each value to its (x, y) cell-center coordinate
    # If values repeat (THRML relaxed), use last occurrence
    coords = {}
    for i in range(N):
        for j in range(N):
            v = int(grid[i, j])
            coords[v] = (j, N - 1 - i)

    present = sorted(coords.keys())
    if len(present) < 2:
        ax.axis("off")
        return fig

    # Build path through 1..81 (skip missing values gracefully)
    path = [coords[v] for v in present]
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # Gradient line
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=1.2, alpha=0.85, zorder=2)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    # Node dots
    ax.scatter(xs, ys, c=np.linspace(0, 1, len(xs)),
               cmap="plasma", s=14, zorder=3, alpha=0.7)

    # Start (white) and end (gold)
    ax.scatter([xs[0]], [ys[0]], c="white", s=55, zorder=5, edgecolors="white")
    ax.scatter([xs[-1]], [ys[-1]], c="gold", s=55, zorder=5, edgecolors="gold")

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def render_energy(history: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xs = list(range(len(history)))
    ax.plot(xs, history, color=ACCENT, linewidth=1.5, alpha=0.9)
    ax.fill_between(xs, history, alpha=0.15, color=ACCENT)
    ax.axhline(0, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel("step (sampled)", color="gray", fontsize=8)
    ax.set_ylabel("energy", color="gray", fontsize=8)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    if history:
        ax.set_title(f"E = {history[-1]:.0f}  (best = {min(history):.0f})",
                     color="white", fontsize=9, pad=4)

    fig.tight_layout(pad=0.4)
    return fig


# ── Streamlit layout ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Lunar Magic Square", page_icon="🌙", layout="wide")
st.markdown(
    "<h1 style='color:#a78bfa;margin-bottom:0'>🌙 Lunar Magic Square</h1>"
    "<p style='color:#6b7280;margin-top:2px'>Live cooling — watch the grid find harmony</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Sampler")
    mode = st.selectbox("Mode", ["Simulated Annealing", "THRML (relaxed)"], label_visibility="collapsed")
    st.markdown("### Parameters")
    steps = st.slider("Steps", 5_000, 300_000, 50_000, step=5_000)
    seed = st.number_input("Seed", value=7, min_value=0, step=1)
    if mode == "Simulated Annealing":
        temp_start = st.slider("Temp start", 0.5, 20.0, 5.0, step=0.5)
        temp_end = st.slider("Temp end", 0.001, 2.0, 0.1, step=0.01)
        diagonal = st.checkbox("Enforce diagonals")
        lunar_weight = st.slider("Lunar weight", 0.0, 10.0, 0.0, step=0.5)
    else:
        chunk_size = st.slider("Chunk size (steps/update)", 50, 500, 150, step=50)
        st.caption("First chunk takes ~15s to JIT-compile. Subsequent chunks are fast.")
        temp_start = temp_end = 5.0
        diagonal = False
        lunar_weight = 0.0

    run = st.button("▶  Run", width="stretch", type="primary")
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>"
        "<b>SA</b>: swap two cells → permutation always valid → true magic squares<br><br>"
        "<b>THRML relaxed</b>: Gibbs-resample each cell freely → values may repeat → "
        "finds balanced grids where every row + col = 369"
        "</small>",
        unsafe_allow_html=True,
    )

col_grid, col_sigil = st.columns([1, 1])
with col_grid:
    st.markdown("**Grid** <small style='color:#6b7280'>— green sums = 369</small>", unsafe_allow_html=True)
    grid_ph = st.empty()
with col_sigil:
    st.markdown("**Sigil** <small style='color:#6b7280'>— path 1 → 81 through the grid</small>", unsafe_allow_html=True)
    sigil_ph = st.empty()

st.markdown("**Energy**")
energy_ph = st.empty()
status_ph = st.empty()

# ── Run loop ──────────────────────────────────────────────────────────────────

if run:
    energy_hist = []

    if mode == "Simulated Annealing":
        gen = sa_generator(steps, seed, temp_start, temp_end, diagonal, lunar_weight)
    else:
        with st.spinner("Building THRML factor graph and JIT-compiling… (~15s first run)"):
            gen = thrml_relaxed_generator(steps, seed, chunk_size)
            # Advance one chunk to trigger compilation inside the spinner
            first = next(gen)

        grid, e, best_e, step, accept, Tcur = first
        energy_hist.append(e)

        fig = render_grid(grid)
        grid_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        fig = render_sigil(grid)
        sigil_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        fig = render_energy(energy_hist)
        energy_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        status_ph.markdown(
            f"<small style='color:#6b7280'>step {step:,} &nbsp;|&nbsp; E = {e:.0f}</small>",
            unsafe_allow_html=True,
        )

    for grid, e, best_e, step, accept, Tcur in gen:
        energy_hist.append(e)

        fig = render_grid(grid)
        grid_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        fig = render_sigil(grid)
        sigil_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        fig = render_energy(energy_hist)
        energy_ph.pyplot(fig, width="stretch")
        plt.close(fig)

        if mode == "Simulated Annealing":
            status_ph.markdown(
                f"<small style='color:#6b7280'>"
                f"step {step:,} &nbsp;|&nbsp; E = {e:.0f} &nbsp;|&nbsp; best = {best_e:.0f} "
                f"&nbsp;|&nbsp; T = {Tcur:.3f} &nbsp;|&nbsp; accept = {accept:.1%}"
                f"</small>",
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                f"<small style='color:#6b7280'>step {step:,} &nbsp;|&nbsp; E = {e:.0f}</small>",
                unsafe_allow_html=True,
            )

    status_ph.markdown(
        f"<b style='color:#4ade80'>✓ Done — best energy: {min(energy_hist):.0f}</b>",
        unsafe_allow_html=True,
    )
