"""Synthetic signals that reproduce the catalog's known failure modes.

Not speech. A harmonic source with a falling spectrum, syllable-rate amplitude
modulation and pauses: enough to exercise bandwidth, silence, clipping and SNR
measurements without shipping audio in the repo.
"""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

RNG = np.random.default_rng(0)


def speechlike(sr=44100, dur=30.0, f0=120.0, noise_dbfs=-75.0):
    t = np.arange(int(sr * dur)) / sr
    f0_t = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0_t) / sr
    x = np.zeros_like(t)
    k = 1
    while k * f0 * 1.05 < sr / 2 * 0.95:
        x += np.sin(k * phase) / k ** 0.8          # gently falling spectrum
        k += 1
    breath = sosfilt(butter(2, 2000, "high", fs=sr, output="sos"), RNG.standard_normal(len(t)))
    x += 0.3 * breath                                # fricative-ish HF content
    env = 0.3 + 0.7 * (0.5 * (1 + np.sin(2 * np.pi * 4 * t)))  # ~4 syllables/sec
    gate = (np.sin(2 * np.pi * 0.25 * t) > -0.8)      # short pause every 4s
    x *= env * gate
    x = 0.3 * x / np.max(np.abs(x))
    return x + 10 ** (noise_dbfs / 20) * RNG.standard_normal(len(t)), sr


def telephone(x, sr, hiss_dbfs=None):
    """Down to 8 kHz, 300-3400 Hz band, mu-law, back up to sr.
    hiss_dbfs adds broadband noise after upsampling (tape digitization case)."""
    y = resample_poly(x, 8000, sr)
    y = sosfilt(butter(4, [300, 3400], "band", fs=8000, output="sos"), y)
    mu = 255
    y = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    y = np.round(y * 127) / 127
    y = np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu
    y = resample_poly(y, sr, 8000)[: len(x)]
    if hiss_dbfs is not None:
        y = y + 10 ** (hiss_dbfs / 20) * RNG.standard_normal(len(y))
    return y


def clipped(x, gain=8.0):
    return np.clip(x * gain, -1.0, 1.0)


def noisy(x, noise_dbfs=-25.0):
    return x + 10 ** (noise_dbfs / 20) * RNG.standard_normal(len(x))


def mostly_silent(x, sr, keep_sec=8.0):
    y = x.copy()
    y[int(keep_sec * sr):] = 10 ** (-80 / 20) * RNG.standard_normal(len(y) - int(keep_sec * sr))
    return y


def lowpassed(x, sr, cutoff=16000):
    """Lossy-codec style lowpass (e.g. 128 kbps MP3 cuts near 16 kHz)."""
    return sosfilt(butter(12, cutoff, "low", fs=sr, output="sos"), x)
