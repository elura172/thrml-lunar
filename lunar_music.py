#!/usr/bin/env python3
"""
Lunar Music — hear a magic square grid as sound.

Each value 1–81 maps to a frequency across three octaves (A2–A5, 110–880 Hz)
using an exponential (musical) scale:

    f(n) = 110 × 2^((n-1)/27)

The 81-voice chord is the "frozen sound" of the cooled lattice.

Usage
-----
  python lunar_music.py                        # plays runs/lunar_magic_9x9.csv (4s)
  python lunar_music.py path/to/grid.csv       # play a specific grid
  python lunar_music.py --duration 8.0         # longer chord
  python lunar_music.py --save chord.wav       # save to WAV instead of playing
  python lunar_music.py --rows                 # scan rows 1–9 as a melodic passage

Dependencies
------------
  pip install sounddevice   # only needed for live playback (not --save)
"""

import argparse
import io
import sys
import wave

import numpy as np

SAMPLERATE = 44100


def freqs_from_grid(grid: np.ndarray) -> np.ndarray:
    """Map values 1–81 → frequencies 110–880 Hz (3 octaves, exponential)."""
    return 110.0 * (2.0 ** ((grid.flatten() - 1) / 27.0))


def render_chord(grid: np.ndarray, duration: float = 4.0) -> np.ndarray:
    """All 81 cells as simultaneous sine waves."""
    freqs = freqs_from_grid(grid)
    t = np.linspace(0, duration, int(SAMPLERATE * duration), endpoint=False)
    signal = np.sum(np.sin(2 * np.pi * freqs[:, None] * t[None, :]), axis=0)
    peak = np.max(np.abs(signal))
    return (signal / peak * 0.8) if peak > 0 else signal


def render_row_scan(grid: np.ndarray, row_duration: float = 0.5) -> np.ndarray:
    """
    Sweep through the 9 rows as a melodic passage.
    Each row plays as a 9-voice chord for row_duration seconds.
    Total length = 9 × row_duration.
    """
    frames = []
    t = np.linspace(0, row_duration, int(SAMPLERATE * row_duration), endpoint=False)
    # Fade in/out per row to avoid clicks
    fade = np.ones_like(t)
    fade_len = int(SAMPLERATE * 0.02)
    fade[:fade_len] = np.linspace(0, 1, fade_len)
    fade[-fade_len:] = np.linspace(1, 0, fade_len)

    for row in grid:
        freqs = 110.0 * (2.0 ** ((row - 1) / 27.0))
        frame = np.sum(np.sin(2 * np.pi * freqs[:, None] * t[None, :]), axis=0) * fade
        peak = np.max(np.abs(frame))
        frames.append((frame / peak * 0.8) if peak > 0 else frame)

    return np.concatenate(frames)


def signal_to_wav_bytes(signal: np.ndarray) -> bytes:
    s16 = (signal * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes(s16.tobytes())
    return buf.getvalue()


def save_wav(signal: np.ndarray, path: str):
    with open(path, "wb") as f:
        f.write(signal_to_wav_bytes(signal))
    print(f"[saved] {path}")


def play_signal(signal: np.ndarray):
    try:
        import sounddevice as sd
    except ImportError:
        print("[error] sounddevice not installed — run: pip install sounddevice")
        print("        Or use --save <path.wav> to save instead of playing.")
        sys.exit(1)
    sd.play(signal.astype(np.float32), SAMPLERATE)
    sd.wait()


def main():
    p = argparse.ArgumentParser(description="Play a magic square grid as sound")
    p.add_argument("csv", nargs="?", default="./runs/lunar_magic_9x9.csv",
                   help="CSV grid file (default: runs/lunar_magic_9x9.csv)")
    p.add_argument("--duration", type=float, default=4.0,
                   help="chord duration in seconds (default 4.0, ignored with --rows)")
    p.add_argument("--rows", action="store_true",
                   help="scan rows 1–9 as a melodic passage instead of full chord")
    p.add_argument("--row-duration", type=float, default=0.5,
                   help="duration per row in --rows mode (default 0.5)")
    p.add_argument("--save", type=str, default=None,
                   help="save to this WAV file instead of playing")
    args = p.parse_args()

    grid = np.loadtxt(args.csv, delimiter=",")
    row_sums = grid.sum(axis=1).astype(int).tolist()
    col_sums = grid.sum(axis=0).astype(int).tolist()
    print(f"Grid : {args.csv}")
    print(f"Rows : {row_sums}")
    print(f"Cols : {col_sums}")

    if args.rows:
        print(f"Mode : row scan (9 rows × {args.row_duration}s each)")
        signal = render_row_scan(grid, row_duration=args.row_duration)
    else:
        print(f"Mode : full chord ({args.duration}s, 81 voices)")
        signal = render_chord(grid, duration=args.duration)

    if args.save:
        save_wav(signal, args.save)
    else:
        print("Playing… (Ctrl-C to stop)")
        play_signal(signal)


if __name__ == "__main__":
    main()
