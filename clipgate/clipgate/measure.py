"""Signal-level measurements for one audio clip.

Everything here is numpy/scipy only and runs at roughly 100x+ realtime on a CPU,
so it can sit early in the pipeline, before any model-based stage.
"""
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

FRAME_SEC = 0.025
HOP_SEC = 0.010
EPS = 1e-12


@dataclass
class ClipMetrics:
    path: str
    declared_sr: int
    channels: int
    duration_sec: float
    effective_bw_hz: float      # highest frequency with real content
    bw_ratio: float             # effective_bw / nyquist. ~1 = uses the band it claims
    rms_dbfs: float
    clip_fraction: float        # fraction of samples at/near full scale
    silence_fraction: float     # fraction of 25ms frames below the silence threshold
    snr_db_est: float           # crude: loud-frame energy vs quiet-frame energy
    dc_offset: float

    def to_dict(self):
        return asdict(self)


def load_mono(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    return x.mean(axis=1), sr, x.shape[1]


def effective_bandwidth(x, sr, win_hz=1000.0, min_step_db=15.0, smooth_hz=200.0):
    """Frequency of the sharpest drop ("cliff") in the long-term spectrum.
    Returns nyquist if no drop is steeper than min_step_db.

    Why a cliff and not a level threshold: telephone audio upsampled to 44.1kHz
    has nothing above ~3.4-4kHz. Upsampled digitally, the top of the band is
    near digital silence. Digitized from tape, the top of the band is flat
    hiss at a level that can sit above real content in a quiet wideband clip.
    Both cases share a steep step at the band edge; natural speech/room
    spectra roll off gradually. A step detector catches both, an absolute
    threshold misses the tape case. Same logic finds MP3 lowpass at ~16kHz.
    """
    nper = min(8192, len(x))
    f, pxx = welch(x, fs=sr, nperseg=nper)
    df = f[1] - f[0]
    # average in the power domain over ~200 Hz: fills the gaps between harmonics
    k = max(1, int(smooth_hz / df))
    level = 10 * np.log10(uniform_filter1d(pxx, k, mode="nearest") + EPS)

    w = max(1, int(win_hz / df))
    best_step, best_i = 0.0, None
    for i in range(w, len(f) - w):
        step = np.median(level[i - w:i]) - np.median(level[i:i + w])
        if step > best_step:
            best_step, best_i = step, i
    if best_i is None or best_step < min_step_db:
        return float(sr / 2)
    return float(f[best_i])


def frame_db(x, sr):
    n = int(FRAME_SEC * sr)
    h = int(HOP_SEC * sr)
    if len(x) < n:
        return np.array([10 * np.log10(np.mean(x ** 2) + EPS)])
    idx = np.arange(0, len(x) - n + 1, h)
    frames = np.stack([x[i:i + n] for i in idx])
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + EPS)


def measure(path, silence_dbfs=-50.0, clip_level=0.999):
    x, sr, ch = load_mono(path)
    db = frame_db(x, sr)
    bw = effective_bandwidth(x, sr)
    return ClipMetrics(
        path=str(path),
        declared_sr=int(sr),
        channels=int(ch),
        duration_sec=round(len(x) / sr, 3),
        effective_bw_hz=round(bw, 1),
        bw_ratio=round(bw / (sr / 2), 3),
        rms_dbfs=round(float(10 * np.log10(np.mean(x ** 2) + EPS)), 2),
        clip_fraction=round(float(np.mean(np.abs(x) >= clip_level)), 5),
        silence_fraction=round(float(np.mean(db < silence_dbfs)), 3),
        # p95 frame ~ speech, p10 frame ~ background. Not a real SNR; a ranking signal.
        snr_db_est=round(float(np.percentile(db, 95) - np.percentile(db, 10)), 2),
        dc_offset=round(float(np.mean(x)), 5),
    )
