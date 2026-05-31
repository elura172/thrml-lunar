"""
Lunar Magic Square — Live Cooling UI
Grid, sigil, and waveform field all update in real time.

Two samplers:
  Simulated Annealing  — swap-based, permutation-preserving, true magic squares
  THRML (relaxed)      — block Gibbs via JAX, uniqueness dropped, values may repeat

Eight waveform fields — each is an energy landscape the sampler cools into:
  None, Row gradient, Column gradient, Radial, Diagonal ↘, Diagonal ↗,
  2D Interference, Lunar wave, Noise

Energy scaling (so field_weight=1 is actually competitive with magic-sum constraint):
  SA:    FIELD_SCALE = 1000  → field_weight 0–5 spans "barely felt" to "dominant"
  THRML: FIELD_SCALE = 17   → same slider, comparable magnitude to unary factors

Run:
  streamlit run app.py
"""

import io
import itertools
import math
import wave

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
FIELD_SCALE_SA = 1000.0    # makes field_weight=1 ~15% of magic energy at random start
FIELD_SCALE_THRML = 17.0   # makes field_weight=1 ~1:1 with THRML unary factors


# ── Waveform fields ───────────────────────────────────────────────────────────

FIELD_MODES = [
    "None",
    "Lunar phases",       # 9 phases mapped row-by-row, symbolically grounded
    "Row gradient",
    "Column gradient",
    "Radial",
    "Diagonal ↘",
    "Diagonal ↗",
    "2D Interference",
    "Lunar wave",
    "Noise",
]

LUNAR_PHASES = [
    ("New Moon",        0.00),
    ("Crescent Waxing", 0.15),
    ("First Quarter",   0.30),
    ("Gibbous Waxing",  0.50),
    ("Full Moon",       1.00),
    ("Gibbous Waning",  0.60),
    ("Last Quarter",    0.40),
    ("Crescent Waning", 0.20),
    ("Dark Moon",       0.05),
]


def make_field(mode: str, seed: int = 7, freq: float = 1.0) -> np.ndarray | None:
    """Return a (9,9) float32 array in [0,1], or None if mode is 'None'."""
    if mode == "None":
        return None

    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    center = (N - 1) / 2.0

    if mode == "Lunar phases":
        intensities = np.array([v for _, v in LUNAR_PHASES], dtype=np.float32)
        field = np.outer(intensities, np.ones(N))

    elif mode == "Row gradient":
        field = np.outer(np.linspace(0, 1, N), np.ones(N))

    elif mode == "Column gradient":
        field = np.outer(np.ones(N), np.linspace(0, 1, N))

    elif mode == "Radial":
        dist = np.sqrt((I - center) ** 2 + (J - center) ** 2)
        field = 1.0 - dist / dist.max()

    elif mode == "Diagonal ↘":
        field = (I + J) / (2.0 * (N - 1))

    elif mode == "Diagonal ↗":
        field = (I + (N - 1 - J)) / (2.0 * (N - 1))

    elif mode == "2D Interference":
        raw = np.sin(freq * np.pi * I / (N - 1)) * np.cos(freq * np.pi * J / (N - 1))
        lo, hi = raw.min(), raw.max()
        field = (raw - lo) / (hi - lo + 1e-8)

    elif mode == "Lunar wave":
        # new→full→new across rows, col modulation by freq
        phase = np.sin(np.pi * np.linspace(0, 1, N))
        decan = np.cos(freq * np.pi * np.linspace(0, 1, N))
        raw = np.outer(phase, decan)
        lo, hi = raw.min(), raw.max()
        field = (raw - lo) / (hi - lo + 1e-8)

    elif mode == "Noise":
        rng = np.random.default_rng(seed)
        field = rng.random((N, N)).astype(np.float32)
        return field

    else:
        return None

    return field.astype(np.float32)


# ── Energy ────────────────────────────────────────────────────────────────────

def compute_energy(
    grid: np.ndarray,
    diagonal: bool = False,
    field: np.ndarray | None = None,
    field_weight: float = 0.0,
) -> float:
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    e = float(np.sum((row_sums - MAGIC_SUM) ** 2) + np.sum((col_sums - MAGIC_SUM) ** 2))

    if diagonal:
        e += float(
            (np.trace(grid) - MAGIC_SUM) ** 2
            + (np.trace(np.fliplr(grid)) - MAGIC_SUM) ** 2
        )

    if field is not None and field_weight > 0.0:
        g_norm = (grid - 1.0) / (N * N - 1)
        e += FIELD_SCALE_SA * field_weight * float(np.sum((g_norm - field) ** 2))

    return e


# ── Generators ────────────────────────────────────────────────────────────────

def sa_generator(
    steps, seed, temp_start, temp_end, diagonal,
    field, field_weight,
    yield_every=300,
):
    rng = np.random.default_rng(seed)
    grid = rng.permutation(DOMAIN).reshape(N, N)
    e = compute_energy(grid, diagonal, field, field_weight)
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
        e_new = compute_energy(grid, diagonal, field, field_weight)
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


def thrml_relaxed_generator(steps, seed, chunk_size, field, field_weight):
    """
    THRML Option A — relaxed uniqueness, field wired into unary factor weights.
    """
    import jax
    import jax.numpy as jnp
    from thrml import (
        CategoricalNode, Block, BlockGibbsSpec, SamplingSchedule, sample_states,
    )
    from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional
    from thrml.factor import FactorSamplingProgram

    K = N * N
    vals = np.arange(1, K + 1, dtype=np.float32)
    key = jax.random.PRNGKey(seed)

    grid_nodes = [[CategoricalNode() for _ in range(N)] for _ in range(N)]
    flat_nodes = [grid_nodes[i][j] for i in range(N) for j in range(N)]

    # Unary: W[cell, v] = 2*(2·M·v - v²)
    w_unary = np.tile(2.0 * (2.0 * MAGIC_SUM * vals - vals ** 2), (K, 1))  # [81, 81]

    # Wire field into unary weights: W_field[cell, val] = -scale*(val - target)²
    if field is not None and field_weight > 0.0:
        targets = field.flatten() * (K - 1) + 1   # target value per cell, [81]
        w_field = -FIELD_SCALE_THRML * field_weight * (
            vals[None, :] - targets[:, None]
        ) ** 2
        w_unary = w_unary + w_field

    factors = []
    factors.append(CategoricalEBMFactor(
        node_groups=[Block(flat_nodes)],
        weights=jnp.array(w_unary),
    ))

    # Pairwise rows + cols
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

    from thrml.factor import FactorSamplingProgram
    program = FactorSamplingProgram(gibbs_spec, samplers, factors, [])

    rng = np.random.default_rng(seed)
    init_indices = rng.integers(0, K, size=K).astype(np.uint8)
    current_state = [jnp.array([v]) for v in init_indices]

    schedule = SamplingSchedule(n_warmup=chunk_size, n_samples=1, steps_per_sample=1)

    for chunk in range(max(1, steps // chunk_size)):
        key, subkey = jax.random.split(key)
        results = sample_states(subkey, program, schedule, current_state, [], free_blocks)
        current_state = [r[-1] for r in results]

        indices = np.array([int(r[-1, 0]) for r in results], dtype=int)
        grid = (indices + 1).reshape(N, N)
        e = compute_energy(grid, field=field, field_weight=field_weight)

        yield grid, e, e, (chunk + 1) * chunk_size, 0.0, 0.0


# ── Renderers ─────────────────────────────────────────────────────────────────

BG = "#0e1117"
ACCENT = "#a78bfa"


def render_grid(grid: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.imshow(grid, cmap="plasma", vmin=1, vmax=81, aspect="equal")

    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(grid[i, j]),
                    ha="center", va="center", color="white",
                    fontsize=7.5, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    for i, rs in enumerate(row_sums):
        ax.text(N - 0.5 + 0.7, i, f"{rs}",
                ha="left", va="center", fontsize=7, fontweight="bold",
                color="#4ade80" if rs == MAGIC_SUM else "#f87171")
    for j, cs in enumerate(col_sums):
        ax.text(j, N - 0.5 + 0.6, f"{cs}",
                ha="center", va="top", fontsize=7, fontweight="bold",
                color="#4ade80" if cs == MAGIC_SUM else "#f87171")

    ax.set_xlim(-0.5, N + 0.8)
    ax.set_ylim(N + 0.9, -0.5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def render_sigil(grid: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    coords = {}
    for i in range(N):
        for j in range(N):
            coords[int(grid[i, j])] = (j, N - 1 - i)

    present = sorted(coords.keys())
    if len(present) < 2:
        ax.axis("off")
        return fig

    path = [coords[v] for v in present]
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=1.2, alpha=0.85, zorder=2)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)

    ax.scatter(xs, ys, c=np.linspace(0, 1, len(xs)),
               cmap="plasma", s=14, zorder=3, alpha=0.7)
    ax.scatter([xs[0]], [ys[0]], c="white", s=55, zorder=5)
    ax.scatter([xs[-1]], [ys[-1]], c="gold", s=55, zorder=5)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def render_field(field: np.ndarray, mode: str, field_weight: float) -> plt.Figure:
    """
    Field preview: heatmap of the waveform intensity, annotated with
    the target value each cell is being pulled toward.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor(BG)

    # Left: raw field intensity
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.imshow(field, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_title("intensity", color="#9ca3af", fontsize=7, pad=3)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{field[i,j]:.1f}",
                    ha="center", va="center", color="white", fontsize=4.5, alpha=0.9)
    # Annotate phase names for Lunar phases mode
    if mode == "Lunar phases":
        for i, (name, _) in enumerate(LUNAR_PHASES):
            ax.text(-0.6, i, name, ha="right", va="center",
                    color="#a78bfa", fontsize=4, style="italic")
    ax.axis("off")

    # Right: target values (what number the annealer wants here)
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    targets = (field * 80 + 1).astype(int)
    ax2.imshow(targets, cmap="plasma", vmin=1, vmax=81, aspect="equal")
    ax2.set_title("target value", color="#9ca3af", fontsize=7, pad=3)
    for i in range(N):
        for j in range(N):
            ax2.text(j, i, str(targets[i, j]),
                     ha="center", va="center", color="white", fontsize=4.5, alpha=0.9)
    ax2.axis("off")

    fig.suptitle(
        f"{mode}  (weight {field_weight:.1f})",
        color=ACCENT, fontsize=8, y=1.01,
    )
    fig.tight_layout(pad=0.3)
    return fig


def render_energy(history: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xs = list(range(len(history)))
    ax.plot(xs, history, color=ACCENT, linewidth=1.5, alpha=0.9)
    ax.fill_between(xs, history, alpha=0.15, color=ACCENT)
    ax.axhline(0, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("update", color="#6b7280", fontsize=8)
    ax.set_ylabel("energy", color="#6b7280", fontsize=8)
    ax.tick_params(colors="#6b7280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    if history:
        ax.set_title(
            f"E = {history[-1]:.0f}   best = {min(history):.0f}",
            color="white", fontsize=9, pad=4,
        )
    fig.tight_layout(pad=0.4)
    return fig


# ── Audio ─────────────────────────────────────────────────────────────────────

SAMPLERATE = 44100


def grids_to_wav_bytes(
    grid_history: list,
    frame_duration: float = 0.2,
) -> bytes:
    """
    Convert a list of grids to a WAV audio stream.
    Each grid becomes one frame of 81 simultaneous sine waves (one per cell).
    Values 1–81 → frequencies 110–880 Hz (3 octaves, exponential).
    """
    t = np.linspace(0, frame_duration, int(SAMPLERATE * frame_duration), endpoint=False)
    # Fade in/out to avoid clicks between frames
    fade = np.ones_like(t)
    fade_len = int(SAMPLERATE * 0.015)
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)

    frames = []
    for grid in grid_history:
        freqs = 110.0 * (2.0 ** ((grid.flatten() - 1) / 27.0))
        frame = np.sum(np.sin(2 * np.pi * freqs[:, None] * t[None, :]), axis=0) * fade
        peak = np.max(np.abs(frame))
        frames.append((frame / peak * 0.7) if peak > 0 else frame)

    signal = np.concatenate(frames)
    s16 = (signal * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes(s16.tobytes())
    return buf.getvalue()


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Lunar Magic Square", page_icon="🌙", layout="wide")
st.markdown(
    "<h1 style='color:#a78bfa;margin-bottom:0'>🌙 Lunar Magic Square</h1>"
    "<p style='color:#6b7280;margin-top:2px'>Live cooling — choose a waveform field and watch the grid settle into it</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Sampler")
    mode = st.selectbox("Mode", ["Simulated Annealing", "THRML (relaxed)"],
                        label_visibility="collapsed")
    steps = st.slider("Steps", 5_000, 300_000, 50_000, step=5_000)
    seed  = st.number_input("Seed", value=7, min_value=0, step=1)

    if mode == "Simulated Annealing":
        temp_start = st.slider("Temp start", 0.5, 20.0, 5.0, step=0.5)
        temp_end   = st.slider("Temp end",   0.001, 2.0, 0.1, step=0.01)
        diagonal   = st.checkbox("Enforce diagonals")
        chunk_size = 150
    else:
        temp_start = temp_end = 5.0
        diagonal = False
        chunk_size = st.slider("Chunk size (steps/update)", 50, 500, 150, step=50)
        st.caption("First chunk compiles JAX graph (~15s). Subsequent chunks are fast.")

    st.markdown("---")
    st.markdown("### Waveform field")
    field_mode   = st.selectbox("Field", FIELD_MODES)
    field_weight = st.slider("Field weight", 0.0, 5.0, 1.0, step=0.1,
                             help="0 = pure magic square  ·  5 = field dominates")

    freq = 1.0
    if field_mode in ("2D Interference", "Lunar wave"):
        freq = st.slider("Frequency", 0.5, 4.0, 1.0, step=0.25)

    st.markdown("---")
    run = st.button("▶  Run", width="stretch", type="primary")
    st.markdown(
        "<small style='color:#6b7280'>"
        "<b>SA</b>: swap two cells → permutation always valid → true magic squares<br><br>"
        "<b>THRML relaxed</b>: Gibbs-resample each cell → values may repeat → "
        "rows + cols → 369, shaped by the field<br><br>"
        "<b>Field weight 0</b>: pure magic constraint<br>"
        "<b>Field weight 5</b>: field dominates, magic is secondary"
        "</small>",
        unsafe_allow_html=True,
    )

# Compute field immediately (live preview, updates on any widget change)
active_field = make_field(field_mode, int(seed), freq)

# ── Main area ─────────────────────────────────────────────────────────────────

col_grid, col_sigil = st.columns([1, 1])
with col_grid:
    st.markdown("**Grid** <small style='color:#6b7280'>— green = 369</small>",
                unsafe_allow_html=True)
    grid_ph = st.empty()
with col_sigil:
    st.markdown("**Sigil** <small style='color:#6b7280'>— path 1 → 81 through the grid</small>",
                unsafe_allow_html=True)
    sigil_ph = st.empty()

# Field preview — always visible, updates instantly with sidebar controls
if active_field is not None:
    st.markdown(
        f"**Waveform field** <small style='color:#6b7280'>— {field_mode} · "
        f"weight {field_weight:.1f} · left: intensity · right: target value per cell</small>",
        unsafe_allow_html=True,
    )
    fig = render_field(active_field, field_mode, field_weight)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
else:
    st.markdown(
        "<small style='color:#444'>No waveform field — pure magic-square annealing. "
        "Select a field above to add a gravitational bias.</small>",
        unsafe_allow_html=True,
    )

st.markdown("**Energy**")
energy_ph = st.empty()
status_ph  = st.empty()

# ── Run loop ──────────────────────────────────────────────────────────────────

if run:
    energy_hist = []
    grid_history = []   # for audio: sampled grid states across the run
    energy_field = active_field if (active_field is not None and field_weight > 0) else None

    if mode == "Simulated Annealing":
        gen = sa_generator(
            steps, int(seed), temp_start, temp_end, diagonal,
            energy_field, field_weight,
        )
    else:
        with st.spinner("Building THRML factor graph and JIT-compiling… (~15s first run)"):
            gen = thrml_relaxed_generator(steps, int(seed), chunk_size, energy_field, field_weight)
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

    for grid, e, best_e, step, accept, Tcur in gen:
        energy_hist.append(e)
        if len(grid_history) < 120:   # cap at 120 frames (~24s of audio at 0.2s/frame)
            grid_history.append(grid.copy())

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
                f"step {step:,} &nbsp;|&nbsp; E = {e:.0f} &nbsp;|&nbsp; "
                f"best = {best_e:.0f} &nbsp;|&nbsp; T = {Tcur:.3f} &nbsp;|&nbsp; "
                f"accept = {accept:.1%}"
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

    if grid_history:
        st.markdown("---")
        st.markdown(
            "**🎵 Listen to the annealing** "
            "<small style='color:#6b7280'>— each grid snapshot becomes a chord of 81 sine waves "
            "(values 1–81 → 110–880 Hz, three octaves)</small>",
            unsafe_allow_html=True,
        )
        with st.spinner("Rendering audio…"):
            wav_bytes = grids_to_wav_bytes(grid_history, frame_duration=0.2)
        duration_s = len(grid_history) * 0.2
        st.audio(wav_bytes, format="audio/wav")
        st.caption(
            f"{len(grid_history)} frames · {duration_s:.0f}s · "
            f"early frames = chaos, late frames = harmony · "
            f"tip: also try `python lunar_music.py --rows` for a melodic row scan"
        )
