"""Accuracy per language for the emotion tagger on labelled public clips.

Input: a CSV with columns path,label,language. The label must already be in
our label set (README shows how each dataset maps). Output: accuracy,
unweighted average recall and a confusion matrix per language.

These sets are all in EmoBox, which emotion2vec+ was fine-tuned from. Treat
the numbers as an optimistic upper bound, not accuracy on the catalog.

    python benchmark.py manifest.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import soundfile as sf

from emotag.api import LABELS, UNKNOWN, assign_emotion
from emotag.backends import Emotion2VecScorer


def report(lang, pairs):
    truth = [c for c in LABELS if any(t == c for t, _ in pairs)]
    cols = LABELS + (UNKNOWN,)
    acc = np.mean([t == p for t, p in pairs])
    # every class counts the same, so "neutral for everything" can't score well
    uar = np.mean([np.mean([p == c for t, p in pairs if t == c]) for c in truth])
    print(f"\n{lang}  n={len(pairs)}  accuracy={acc:.3f}  UAR={uar:.3f}")
    print(f"{'true/pred':<10}" + "".join(f"{c[:7]:>9}" for c in cols))
    for t in truth:
        print(f"{t:<10}" + "".join(f"{sum(tt == t and p == c for tt, p in pairs):>9}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()

    scorer = Emotion2VecScorer()
    pairs = defaultdict(list)  # language -> [(true, predicted)]
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            x, sr = sf.read(row["path"], always_2d=True)
            x = x.mean(axis=1)
            # Each file is one acted utterance, often under 1 s (MESD averages
            # under a second). min_seg_s=0 measures the model, not our length rule.
            clip = assign_emotion(x, sr, [(0.0, len(x) / sr)], scorer, min_seg_s=0.0)
            pairs[row["language"]].append((row["label"], clip.label))
    for lang in sorted(pairs):
        report(lang, pairs[lang])


if __name__ == "__main__":
    main()
