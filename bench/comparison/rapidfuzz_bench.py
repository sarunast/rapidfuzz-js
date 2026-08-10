"""The Python leg of the library comparison.

Reads the corpus `run.mjs` wrote, times the same operations with the same loop
shape and the same statistic, and prints seconds-per-pass as JSON on stdout.

Run through `run.mjs`, not directly:

    python3 -m venv .venv && .venv/bin/pip install rapidfuzz
    node bench/comparison/run.mjs --python=.venv/bin/python

The timing here mirrors `timing.mjs` deliberately — three warm-up passes, a
pass scaled to at least 50ms, nine timed ones, report the median. There is less
to warm up in CPython than in V8, but the *shape* has to match or the two
columns are not measuring the same thing, and the scaling matters on both sides
for the same reason: this machine spikes every few minutes, and against a
workload of tens of microseconds a spike is the whole measurement.
`perf_counter` is the clock with the fewest surprises across platforms.
"""

from __future__ import annotations

import json
import statistics
import sys
from time import perf_counter
from typing import Callable

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

WARMUPS = 3
PASSES = 9
TARGET_SECONDS = 0.05


def time_it(run: Callable[[], None]) -> float:
    """Median seconds per single run, after warming up and sizing a pass."""
    for _ in range(WARMUPS):
        run()

    inner = 1
    while True:
        started = perf_counter()
        for _ in range(inner):
            run()
        elapsed = perf_counter() - started
        if elapsed >= TARGET_SECONDS or inner >= 1_000_000:
            break
        factor = max(2, int(TARGET_SECONDS / max(elapsed, 1e-9)) + 1)
        inner = min(1_000_000, inner * factor)

    seconds = []
    for _ in range(PASSES):
        started = perf_counter()
        for _ in range(inner):
            run()
        seconds.append((perf_counter() - started) / inner)
    return statistics.median(seconds)


def main() -> None:
    with open(sys.argv[1], encoding="utf8") as handle:
        corpus = json.load(handle)

    results: dict[str, float] = {}

    for length, pairs in corpus["pairs"].items():
        # Bound to locals: an attribute lookup per iteration is real overhead in
        # CPython, and leaving it in would measure the interpreter rather than
        # the library. The JavaScript side has the same call hoisted for it by
        # the JIT, so this keeps the two comparable.
        distance = Levenshtein.distance
        results[f"levenshtein-{length}"] = time_it(
            lambda pairs=pairs, distance=distance: [distance(a, b) for a, b in pairs]
            and None
        )

    sentences = corpus["sentences"]
    ratio = fuzz.ratio
    results["ratio-sentences"] = time_it(
        lambda: [ratio(a, b) for a, b in sentences] and None
    )

    choices = corpus["choices"]
    queries = corpus["queries"]
    extract_one = process.extractOne
    results["extract-one"] = time_it(
        lambda: [extract_one(q, choices, scorer=ratio) for q in queries] and None
    )

    rows = corpus["matrixRows"]
    cols = corpus["matrixCols"]
    results["score-matrix"] = time_it(
        lambda: process.cdist(rows, cols, scorer=ratio) is None
    )

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
