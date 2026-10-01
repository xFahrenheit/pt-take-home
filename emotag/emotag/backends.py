"""emotion2vec+ large through FunASR, mapped onto our label set.

Why this model: it's trained on speech emotion data across many languages,
outputs a superset of our labels, and runs on CPU (~15x realtime on a laptop).
"""
import numpy as np
from scipy.signal import resample_poly

from .api import LABELS, UNKNOWN

MODEL_SR = 16000  # emotion2vec+ is trained on 16 kHz speech

# The model's 9 classes. "other" and "<unk>" aren't emotions we deliver, so
# they fall through to UNKNOWN.
LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise",
}


class Emotion2VecScorer:
    def __init__(self, model_id="iic/emotion2vec_plus_large", hub="hf"):
        from funasr import AutoModel  # lazy: tests and the rest of the pipeline don't need torch

        self.name = model_id
        self._model = AutoModel(model=model_id, hub=hub, disable_update=True,
                                disable_pbar=True, log_level="ERROR")

    def score(self, audio, sr):
        if sr != MODEL_SR:
            g = np.gcd(sr, MODEL_SR)
            audio = resample_poly(audio, MODEL_SR // g, sr // g)
        res = self._model.generate(audio.astype(np.float32), granularity="utterance",
                                   extract_embedding=False)[0]
        probs = dict.fromkeys(LABELS + (UNKNOWN,), 0.0)
        # raw labels look like "生气/angry"; the last one is "<unk>"
        for raw, p in zip(res["labels"], res["scores"]):
            probs[LABEL_MAP.get(raw.split("/")[-1], UNKNOWN)] += float(p)
        return probs
