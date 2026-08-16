"""Generate the n-gram oracle consumed by tests/parity/ngram.test.ts.

RapidFuzz ships neither Sørensen-Dice nor Cosine, so there is no upstream to port
from and no C++/Python pair to compare. The reference below is therefore written
from the formula in `collections.Counter`, which is short enough to read as the
specification it is.

`textdistance.Sorensen(qval=n, as_set=False)` is multiset Dice and is asserted
against here as an independent second opinion — but only where both sides have
grams, because its early return for inputs shorter than `qval` fires before any
q-gram is built and answers a different question than the formula does. Its
`Cosine` is Otsuka-Ochiai over intersection counts rather than a dot product of
frequency vectors, so it is not usable as an oracle at all and is not imported.
"""

from __future__ import annotations

import json
from collections import Counter
from math import sqrt
from pathlib import Path

import textdistance


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "fixtures" / "ngram-oracle.json"


def profile(sequence, gram_size):
    """The exact multiset of `gram_size`-grams, keyed by the gram itself."""
    return Counter(
        tuple(sequence[start : start + gram_size])
        for start in range(len(sequence) - gram_size + 1)
    )


def dice(left, right, gram_size):
    a = profile(left, gram_size)
    b = profile(right, gram_size)
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 or total_b == 0:
        # No grams on either side means the ratio is 0/0. Two such sequences are
        # as similar as they are equal; against one that does have grams they
        # share none.
        both_empty = total_a == 0 and total_b == 0
        return 1.0 if both_empty and list(left) == list(right) else 0.0
    shared = sum(min(a[gram], b[gram]) for gram in a.keys() & b.keys())
    return 2 * shared / (total_a + total_b)


def cosine(left, right, gram_size):
    a = profile(left, gram_size)
    b = profile(right, gram_size)
    if sum(a.values()) == 0 or sum(b.values()) == 0:
        # `‖A‖ = 0` with `‖B‖ > 0` is `0/0`, not 0, so this branch is needed on
        # both sides — unlike Dice, where one-sided emptiness falls out of the
        # formula.
        both_empty = sum(a.values()) == 0 and sum(b.values()) == 0
        return 1.0 if both_empty and list(left) == list(right) else 0.0
    dot = sum(a[gram] * b[gram] for gram in a.keys() & b.keys())
    norm_a = sum(count * count for count in a.values())
    norm_b = sum(count * count for count in b.values())
    return dot / sqrt(norm_a * norm_b)


def tversky(left, right, gram_size, alpha, beta):
    a = profile(left, gram_size)
    b = profile(right, gram_size)
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 or total_b == 0:
        # The same equality fallback as Dice: no grams on a side leaves nothing
        # for the weights to price.
        both_empty = total_a == 0 and total_b == 0
        return 1.0 if both_empty and list(left) == list(right) else 0.0
    shared = sum(min(a[gram], b[gram]) for gram in a.keys() & b.keys())
    return shared / (
        shared + alpha * (total_a - shared) + beta * (total_b - shared)
    )


PAIRS = [
    ("", ""),
    ("a", "a"),
    ("a", "b"),
    ("a", "ab"),
    ("ab", "ab"),
    ("night", "nacht"),
    ("banana", "bananas"),
    ("abcabcab", "ababab"),
    ("kitten", "sitting"),
    ("the wonderful new york mets", "new york mets"),
    ("South Korea", "North Korea"),
    ("abcdef", "abcdef\U0001f600\U0001f600\U0001f600\U0001f600\U0001f600"),
    ("\U0001f600a", "\U0001f600b"),
    ("\U0001f600\U0001f600", "\U0001f600\U0001f600"),
    ("é", "é"),
    ([1, 2, 3], [1, 4, 3]),
    ([1, 2, 3, 1, 2], [1, 2, 1, 2, 3]),
    (["foo", "bar"], ["foo", "baz"]),
    (["a,b", "c"], ["a", "b,c"]),
]

GRAM_SIZES = [1, 2, 3, 7]

dice_cases = []
cosine_cases = []
for left, right in PAIRS:
    for gram_size in GRAM_SIZES:
        similarity = dice(left, right, gram_size)
        if (
            isinstance(left, str)
            and min(len(left), len(right)) >= gram_size
            and left != right
        ):
            crosschecked = textdistance.Sorensen(
                qval=gram_size, as_set=False
            ).similarity(left, right)
            if abs(crosschecked - similarity) > 1e-12:
                raise SystemExit(
                    f"textdistance disagrees on {left!r}/{right!r} at qval="
                    f"{gram_size}: {crosschecked} vs {similarity}"
                )
        dice_cases.append(
            {
                "left": left,
                "right": right,
                "gramSize": gram_size,
                "similarity": similarity,
            }
        )
        cosine_cases.append(
            {
                "left": left,
                "right": right,
                "gramSize": gram_size,
                "similarity": cosine(left, right, gram_size),
            }
        )

# Tversky is asymmetric once alpha and beta differ, so its cases carry both
# orientations and stay out of the symmetric dice/cosine tables above.
TVERSKY_WEIGHTS = [(0.5, 0.5), (1.0, 1.0), (1.0, 0.0), (0.2, 0.7)]

tversky_cases = []
for left, right in PAIRS:
    for gram_size in [1, 2, 3]:
        for alpha, beta in TVERSKY_WEIGHTS:
            similarity = tversky(left, right, gram_size, alpha, beta)
            if (alpha, beta) == (0.5, 0.5):
                # At the default weights the formula collapses to Dice exactly.
                against_dice = dice(left, right, gram_size)
                if abs(similarity - against_dice) > 1e-15:
                    raise SystemExit(
                        f"tversky(0.5, 0.5) disagrees with dice on "
                        f"{left!r}/{right!r} at gram_size={gram_size}"
                    )
            tversky_cases.append(
                {
                    "left": left,
                    "right": right,
                    "gramSize": gram_size,
                    "alpha": alpha,
                    "beta": beta,
                    "similarity": similarity,
                    "reverseSimilarity": tversky(right, left, gram_size, alpha, beta),
                }
            )

OUTPUT.write_text(
    json.dumps(
        {"dice": dice_cases, "cosine": cosine_cases, "tversky": tversky_cases},
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
print(
    f"wrote {len(dice_cases)} dice, {len(cosine_cases)} cosine and "
    f"{len(tversky_cases)} tversky cases"
)
print(f"  to {OUTPUT.relative_to(ROOT)}")
print("run `pnpm format` — oxfmt collapses the short arrays this writes expanded")
