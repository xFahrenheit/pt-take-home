"""Real ~30 s single-speaker clips through VAD + assign_emotion.

Pure clips: one actor, one emotion, sentences joined until >= 30 s.
Mixed clips: one actor, ~15 s of an emotion then ~15 s of neutral.
Segments come from Silero VAD, as in the pipeline. Run from a folder holding
EMOVO/ and crema/ (CREMA-D AudioWAV files).
"""
import collections
import glob

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

from emotag.api import assign_emotion
from emotag.backends import Emotion2VecScorer

GAP_S = 0.3
vad = load_silero_vad()
scorer = Emotion2VecScorer()


def join(paths, max_s):
    parts, sr, total = [], None, 0.0
    for p in paths:
        x, sr = sf.read(p, always_2d=True)
        parts += [x.mean(axis=1), np.zeros(int(GAP_S * sr))]
        total += len(x) / sr + GAP_S
        if total >= max_s:
            break
    return np.concatenate(parts), sr


def segments(y, sr):
    g = np.gcd(sr, 16000)
    y16 = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    ts = get_speech_timestamps(torch.from_numpy(y16), vad, sampling_rate=16000, return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def run(kind, lang, actor, truth, y, sr):
    segs = segments(y, sr)
    r = assign_emotion(y, sr, segs, scorer)
    seg_labels = "".join(s.label[:2] + " " for s in r.segments)
    print(f"{kind:5} {lang} {actor:5} {truth:8} {len(y)/sr:5.1f}s {len(segs):2}seg -> {r.label:8} "
          f"cons={r.consistency:.2f} conf={r.confidence:.2f} {r.status:14} | {seg_labels}")
    return r


results = []
# Italian: EMOVO, 6 actors x 7 emotions. Files are <emotion>-<actor>-<sentence>.wav
emovo = dict(neu="neutral", gio="happy", tri="sad", rab="angry", pau="fear", sor="surprise", dis="disgust")
for actor in ["f1", "f2", "f3", "m1", "m2", "m3"]:
    for code, lab in emovo.items():
        y, sr = join(sorted(glob.glob(f"EMOVO/{actor}/{code}-{actor}-*.wav")), 30)
        results.append(("pure", "it", lab, run("pure", "it", actor, lab, y, sr)))
    for code in ["rab", "tri", "gio"]:
        a, sr = join(sorted(glob.glob(f"EMOVO/{actor}/{code}-{actor}-*.wav")), 15)
        b, _ = join(sorted(glob.glob(f"EMOVO/{actor}/neu-{actor}-*.wav")), 15)
        results.append(("mixed", "it", f"{emovo[code]}+neu", run("mixed", "it", actor, f"{emovo[code][:3]}+neu", np.concatenate([a, b]), sr)))

# English: CREMA-D actors 1016-1019, 12 sentences per emotion
crema = dict(ANG="angry", SAD="sad", HAP="happy", NEU="neutral")
for actor in ["1016", "1017", "1018", "1019"]:
    for code, lab in crema.items():
        y, sr = join(sorted(glob.glob(f"crema/{actor}_*_{code}_*.wav")), 30)
        results.append(("pure", "en", lab, run("pure", "en", actor, lab, y, sr)))
    for code in ["ANG", "SAD", "HAP"]:
        a, sr = join(sorted(glob.glob(f"crema/{actor}_*_{code}_*.wav"))[:6], 15)
        b, _ = join(sorted(glob.glob(f"crema/{actor}_*_NEU_*.wav"))[:6], 15)
        results.append(("mixed", "en", f"{crema[code]}+neu", run("mixed", "en", actor, f"{crema[code][:3]}+neu", np.concatenate([a, b]), sr)))

print()
for lang in ["it", "en"]:
    pure = [(t, r) for k, l, t, r in results if k == "pure" and l == lang]
    mixed = [(t, r) for k, l, t, r in results if k == "mixed" and l == lang]
    print(f"{lang}: pure n={len(pure)} label correct={sum(r.label == t for t, r in pure)} "
          f"ok and correct={sum(r.label == t and r.status == 'ok' for t, r in pure)} "
          f"status={dict(collections.Counter(r.status for _, r in pure))}")
    print(f"{lang}: mixed n={len(mixed)} flagged mixed={sum(r.status == 'mixed' for _, r in mixed)} "
          f"status={dict(collections.Counter(r.status for _, r in mixed))}")
