import numpy as np
import pytest

from emotag.api import LABELS, LOW_CONFIDENCE, MIXED, NO_SPEECH, OK, UNKNOWN, assign_emotion

SR = 100  # the fake scorer ignores the rate, and a low one keeps arrays tiny
VOCAB = LABELS + (UNKNOWN,)


class FakeScorer:
    """Reads the label off the audio: a sample value of i means VOCAB[i]."""
    name = "fake"

    def __init__(self, prob=0.9):
        self.prob = prob
        self.calls = []  # segment lengths in seconds, to check skipping and splitting

    def score(self, audio, sr):
        self.calls.append(len(audio) / sr)
        return {VOCAB[int(audio[0])]: self.prob}


def clip(*runs):
    """runs: (label, seconds) pairs, laid end to end."""
    return np.concatenate([np.full(int(s * SR), VOCAB.index(lab), dtype=float) for lab, s in runs])


def test_majority_wins():
    audio = clip(("angry", 20), ("neutral", 10))
    r = assign_emotion(audio, SR, [(0, 20), (20, 30)], FakeScorer())
    assert (r.label, r.status) == ("angry", OK)
    assert r.consistency == pytest.approx(2 / 3)


def test_even_split_is_mixed():
    # a tie can never reach 0.6, so it's always mixed
    audio = clip(("angry", 15), ("neutral", 15))
    r = assign_emotion(audio, SR, [(0, 15), (15, 30)], FakeScorer())
    assert r.status == MIXED
    assert r.consistency == pytest.approx(0.5)


def test_short_segments_skipped():
    audio = clip(("sad", 0.5), ("happy", 9.5))
    s = FakeScorer()
    r = assign_emotion(audio, SR, [(0, 0.5), (0.5, 10)], s)
    assert (r.label, r.consistency) == ("happy", 1.0)
    assert s.calls == [9.5]


def test_long_segment_split_evenly():
    s = FakeScorer()
    assign_emotion(clip(("fear", 25)), SR, [(0, 25)], s)
    assert s.calls == pytest.approx([25 / 3] * 3, abs=1 / SR)  # slicing rounds to whole samples


def test_segment_past_end_is_clamped():
    s = FakeScorer()
    r = assign_emotion(clip(("happy", 10)), SR, [(0, 10.3), (10.1, 12)], s)
    assert s.calls == [10.0]
    assert r.segments[-1].end_sec == 10.0


def test_unknown_lowers_consistency():
    audio = clip(("angry", 10), ("unknown", 10))
    r = assign_emotion(audio, SR, [(0, 10), (10, 20)], FakeScorer())
    assert (r.label, r.status) == ("angry", MIXED)
    assert r.consistency == pytest.approx(0.5)


def test_all_unknown():
    r = assign_emotion(clip(("unknown", 10)), SR, [(0, 10)], FakeScorer())
    assert (r.label, r.status) == (UNKNOWN, LOW_CONFIDENCE)


def test_low_confidence():
    r = assign_emotion(clip(("sad", 10)), SR, [(0, 10)], FakeScorer(prob=0.3))
    assert (r.label, r.status) == ("sad", LOW_CONFIDENCE)


def test_no_segments():
    r = assign_emotion(clip(("sad", 10)), SR, [], FakeScorer())
    assert (r.label, r.status, r.segments) == (UNKNOWN, NO_SPEECH, [])
