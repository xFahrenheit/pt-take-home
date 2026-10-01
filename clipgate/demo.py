"""Build test clips from real public audio and run the gate on them.

Source: LibriSpeech utterances and a solo trumpet recording, fetched via
librosa.ex() (librosa's public example data). LibriSpeech is 16 kHz audio;
librosa ships it stored at 22.05 kHz, which is itself a small example of a
file whose sample rate overstates its content.

Writes demo_out/*.wav and demo_out/results.jsonl.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from clipgate import synth
from clipgate.cli import main

OUT = Path("demo_out")
OUT.mkdir(exist_ok=True)
SR = 44100


def to_30s(x):
    n = 30 * SR
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def load(name):
    x, sr = sf.read(librosa.ex(name), dtype="float64")
    return x, sr


# The CLI uses a process pool. On macOS, workers re-import this script, so the
# demo must only run under the main guard or every worker re-runs it.
if __name__ == "__main__":
    speech = np.concatenate([load(k)[0] for k in ("libri1", "libri2", "libri3")])
    speech_sr = load("libri1")[1]                         # 22050
    speech44 = to_30s(resample_poly(speech, SR, speech_sr))
    trumpet, tsr = load("trumpet")
    trumpet44 = to_30s(resample_poly(trumpet, SR, tsr))

    clips = {
        "libri_as_shipped_22k": (to_30s(speech)[: 30 * speech_sr], speech_sr),
        "libri_upsampled_44k": (speech44, SR),
        "libri_phone_digital_44k": (synth.telephone(speech44, SR), SR),
        "libri_phone_tape_44k": (synth.telephone(speech44, SR, hiss_dbfs=-55), SR),
        "libri_clipped_44k": (synth.clipped(speech44, gain=6), SR),
        "trumpet_44k": (trumpet44, SR),
    }
    for name, (y, sr) in clips.items():
        sf.write(OUT / f"{name}.wav", y, sr)

    main([str(OUT), "--out", str(OUT / "results.jsonl")])

    print(f"\n{'clip':28s} {'sr':>6s} {'bw_hz':>7s} {'ratio':>5s} {'snr':>5s}  verdict  reasons")
    for line in open(OUT / "results.jsonl"):
        r = json.loads(line)
        print(f"{Path(r['path']).stem:28s} {r['declared_sr']:6d} {r['effective_bw_hz']:7.0f} "
              f"{r['bw_ratio']:5.2f} {r['snr_db_est']:5.1f}  {r['verdict']:7s}  "
              f"{'; '.join(x.split(':', 2)[1] for x in r['reasons'])}")
