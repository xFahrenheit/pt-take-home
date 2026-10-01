"""python -m clipgate.cli <dir_or_files...> [--out results.jsonl]

Writes one JSON line per clip: metrics + verdict + reasons.
Per-file failures are recorded, not raised, so one bad file can't stop a batch.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .gate import judge
from .measure import measure

EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def run_one(path):
    try:
        m = measure(path)
        verdict, reasons = judge(m)
        return {**m.to_dict(), "verdict": verdict, "reasons": reasons}
    except Exception as e:  # corrupt / unreadable files are data, not crashes
        return {"path": str(path), "verdict": "ERROR", "reasons": [repr(e)]}


def collect(inputs):
    for p in map(Path, inputs):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in EXTS)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="-")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    files = list(collect(a.inputs))
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    counts = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for row in ex.map(run_one, files):
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
            out.write(json.dumps(row) + "\n")
    if out is not sys.stdout:
        out.close()
    print(f"{len(files)} files: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
