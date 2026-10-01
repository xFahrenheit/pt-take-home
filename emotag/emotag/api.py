"""Assign one emotion label to a ~30 s single-speaker clip.

The catalog has no emotion field, so this is the enrichment step in phase 2.
It scores each speech segment on its own and votes, instead of scoring the
whole clip in one pass. 30 s is long for emotion. One pass over the clip
averages a short peak (5 s of anger in 25 s of calm) into "neutral".
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

# Ekman's six plus neutral: the label set most public emotion corpora use
LABELS = ("neutral", "happy", "sad", "angry", "fear", "surprise", "disgust")
UNKNOWN = "unknown"

OK, MIXED, LOW_CONFIDENCE, NO_SPEECH = "ok", "mixed", "low_confidence", "no_speech"


class EmotionScorer(Protocol):
    name: str

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Probabilities over LABELS and UNKNOWN for one mono segment."""


@dataclass
class SegmentEmotion:
    start_sec: float
    end_sec: float
    label: str
    prob: float


@dataclass
class ClipEmotion:
    label: str          # majority label by speech time, UNKNOWN if nothing usable
    consistency: float  # share of scored speech time that carries `label`
    confidence: float   # mean prob of `label` over the segments that voted for it
    status: str
    model: str
    segments: list[SegmentEmotion]

    def to_dict(self):
        return asdict(self)


def _chunks(segments, min_seg_s, max_seg_s):
    for start, end in segments:
        dur = end - start
        if dur < min_seg_s:
            continue
        # equal pieces, so a 12 s segment becomes 2 x 6 s, not 10 s + a 2 s scrap
        n = int(np.ceil(dur / max_seg_s))
        step = dur / n
        for i in range(n):
            yield start + i * step, start + (i + 1) * step


def assign_emotion(audio, sr, segments, scorer: EmotionScorer,
                   min_seg_s=1.0,         # under ~1 s there's too little prosody to judge
                   max_seg_s=10.0,        # emotion models train on utterances a few seconds long
                   min_consistency=0.6,   # a clear majority, not a plurality. Pilot sets the final value
                   min_confidence=0.5,    # the model puts more weight on this label than on all others combined
                   ) -> ClipEmotion:
    """segments: (start_sec, end_sec) speech regions from VAD."""
    # VAD and the decoder can disagree on length at the end of a file. An empty
    # slice crashes the model, so clamp to the audio and let short scraps drop.
    dur = len(audio) / sr
    segments = [(s, min(e, dur)) for s, e in segments]
    scored = []
    for start, end in _chunks(segments, min_seg_s, max_seg_s):
        probs = scorer.score(audio[int(start * sr):int(end * sr)], sr)
        label = max(probs, key=probs.get)
        scored.append(SegmentEmotion(start, end, label if label in LABELS else UNKNOWN, probs[label]))

    if not scored:
        return ClipEmotion(UNKNOWN, 0.0, 0.0, NO_SPEECH, scorer.name, [])

    speech_time = defaultdict(float)
    for s in scored:
        speech_time[s.label] += s.end_sec - s.start_sec
    total = sum(speech_time.values())

    # unknown can't win, but its time stays in `total`, so it lowers consistency
    known = {k: v for k, v in speech_time.items() if k != UNKNOWN}
    if not known:
        return ClipEmotion(UNKNOWN, 0.0, 0.0, LOW_CONFIDENCE, scorer.name, scored)

    label = max(known, key=known.get)
    consistency = known[label] / total
    confidence = sum(s.prob * (s.end_sec - s.start_sec) for s in scored if s.label == label) / known[label]

    if consistency < min_consistency:
        status = MIXED
    elif confidence < min_confidence:
        status = LOW_CONFIDENCE
    else:
        status = OK
    return ClipEmotion(label, consistency, confidence, status, scorer.name, scored)
