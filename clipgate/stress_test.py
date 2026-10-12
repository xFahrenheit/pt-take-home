"""Stress test on real speech: ~30 s single-speaker clips under controlled damage.

Clips are built from public acted emotional speech (EMOVO Italian, CREMA-D
English, CaFE French), one speaker and one emotion per clip. For each clip:
  - versions that aren't band-limited must not read as narrowband
  - simulated telephone versions must not PASS the gate

    python stress_test.py <data_dir>   # holds EMOVO/, crema/, cafe/ (any subset)

The telephone audio is simulated (synth.telephone), not a real archive.
"""
import collections
import glob
import os
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from clipgate import synth
from clipgate.gate import judge
from clipgate.measure import effective_bandwidth, measure

NARROW = 4500.0  # gate.py rejects below this
RNG = np.random.default_rng(1)


def clips(data):
    """(speaker, emotion) -> file paths, from each dataset's filename convention."""
    g = collections.defaultdict(list)
    for p in glob.glob(f"{data}/EMOVO/*/*.wav"):      # dis-f1-b1.wav
        emo, spk = os.path.basename(p).split("-")[:2]
        g[("it", spk, emo)].append(p)
    for p in glob.glob(f"{data}/cafe/*.wav"):         # 01-C-1-1.wav
        spk, emo = os.path.basename(p).split("-")[:2]
        g[("fr", spk, emo)].append(p)
    for p in glob.glob(f"{data}/crema/*.wav"):        # 1001_DFA_ANG_XX.wav
        spk, _, emo = os.path.basename(p).split("_")[:3]
        g[("en", spk, emo)].append(p)
    return g


def noise(x, dbfs):
    return x + 10 ** (dbfs / 20) * RNG.standard_normal(len(x))


def conditions(x, sr):
    s = 10 * np.log10(np.mean(x ** 2) + 1e-12)       # speech level, dBFS
    yield "wide", "original", x, sr
    yield "wide", "quiet_-30dB", noise(x * 10 ** (-30 / 20), -90), sr
    yield "wide", "noise_snr20", noise(x, s - 20), sr
    up = resample_poly(x, 44100, sr)                    # the phone archive is stored at 44.1 kHz
    phone = synth.telephone(up, 44100)
    yield "phone", "phone_digital", phone, 44100
    yield "phone", "phone_tape_hiss_-40dB", noise(phone, s - 40), 44100
    yield "phone", "phone_tape_hiss_-25dB", noise(phone, s - 25), 44100
    yield "phone", "phone_quiet", noise(phone * 10 ** (-30 / 20), -90), 44100


def run(paths):
    parts, sr, dur = [], None, 0.0
    for p in sorted(paths):
        x, sr = sf.read(p, always_2d=True)
        parts.append(x.mean(axis=1))
        dur += len(parts[-1]) / sr
        if dur >= 30:
            break
    if dur < 25:
        return []
    rows = []
    for kind, cond, y, r in conditions(np.concatenate(parts), sr):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, np.clip(y, -1, 1), r)
            verdict, _ = judge(measure(f.name))
        rows.append((kind, cond,
                     effective_bandwidth(y, r, min_edge_hz=1000.0, weak_step_db=15.0) < NARROW,  # as first shipped
                     effective_bandwidth(y, r) < NARROW, verdict))
    return rows


if __name__ == "__main__":
    groups = clips(sys.argv[1])
    with Pool() as pool:
        rows = [r for rs in pool.map(run, groups.values()) for r in rs]
    n_clips = sum(r[1] == "original" for r in rows)
    print(f"{n_clips} clips of 25 to 33 s, one speaker and one emotion each.\n")
    print(f"{'condition':22} {'should be':10} {'narrowband: before':>19} {'now':>5}   gate verdicts now")
    by = collections.defaultdict(list)
    for kind, cond, old, new, v in rows:
        by[(kind, cond)].append((old, new, v))
    for (kind, cond), xs in sorted(by.items(), key=lambda k: (k[0][0] != "wide", k[0][1])):
        v = collections.Counter(x[2] for x in xs)
        print(f"{cond:22} {'narrow' if kind == 'phone' else 'wide':10} "
              f"{sum(x[0] for x in xs):>12}/{len(xs):<6} {sum(x[1] for x in xs):>5}   "
              f"PASS {v['PASS']:>3}  FLAG {v['FLAG']:>3}  REJECT {v['REJECT']:>3}")
