"""Generate the RapidFuzz 3.14.5 oracle consumed by parity tests.

`import rapidfuzz` binds the C++ extension; the `*_py` siblings are the pure-Python
fallback, and the two do not always agree. Cases that agree are stored as one value;
cases that disagree are stored in `divergences` with both, so no case silently
enshrines one backend's artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rapidfuzz
from rapidfuzz import fuzz, fuzz_py, process, process_py, utils
from rapidfuzz.distance import (
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    Levenshtein_py,
    JaroWinkler_py,
    OSA,
    Postfix,
    Prefix,
)


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "3.14.5"
OUTPUT = ROOT / "tests" / "fixtures" / f"rapidfuzz-{EXPECTED_VERSION}.json"

if rapidfuzz.__version__ != EXPECTED_VERSION:
    raise SystemExit(
        f"installed rapidfuzz {rapidfuzz.__version__} would be written to "
        f"{OUTPUT.name}; install {EXPECTED_VERSION} or rename the fixture"
    )

BASE_CASES = [
    ["", ""],
    ["abc", "abc"],
    ["kitten", "sitting"],
    ["CA", "ABC"],
    ["😀a", "😀b"],
    [[1, 2, 3], [1, 4, 3]],
    ["é", "é"],
]


def metric_case(family, module, left, right, configuration=None, python_options=None):
    configuration = configuration or {}
    python_options = python_options or {}
    return {
        "family": family,
        "configuration": configuration,
        "left": left,
        "right": right,
        "scores": {
            "distance": module.distance(left, right, **python_options),
            "similarity": module.similarity(left, right, **python_options),
            "normalizedDistance": module.normalized_distance(
                left, right, **python_options
            ),
            "normalizedSimilarity": module.normalized_similarity(
                left, right, **python_options
            ),
        },
    }


def editop_value(operation):
    return {
        "tag": operation.tag,
        "srcPos": operation.src_pos,
        "destPos": operation.dest_pos,
    }


def opcode_value(operation):
    return {
        "tag": operation.tag,
        "srcStart": operation.src_start,
        "srcEnd": operation.src_end,
        "destStart": operation.dest_start,
        "destEnd": operation.dest_end,
    }


# Upstream spells a matching block `(a, b, size)`; ours names the same three fields.
def matching_block_value(block):
    return {"srcStart": block.a, "destStart": block.b, "length": block.size}


def alignment_value(alignment):
    return {
        "score": alignment.score,
        "srcStart": alignment.src_start,
        "srcEnd": alignment.src_end,
        "destStart": alignment.dest_start,
        "destEnd": alignment.dest_end,
    }


def editops_detail(family, module, left, right, configuration=None, python_options=None):
    configuration = configuration or {}
    python_options = python_options or {}
    edits = module.editops(left, right, **python_options)
    blocks = edits.as_opcodes()
    return {
        "family": family,
        "configuration": configuration,
        "left": left,
        "right": right,
        "srcLen": edits.src_len,
        "destLen": edits.dest_len,
        "editops": [editop_value(value) for value in edits],
        "opcodes": [opcode_value(value) for value in blocks],
        "matchingBlocks": [matching_block_value(b) for b in edits.as_matching_blocks()],
        "opcodeMatchingBlocks": [
            matching_block_value(b) for b in blocks.as_matching_blocks()
        ],
        "inverse": [editop_value(value) for value in edits.inverse()],
        "opcodeInverse": [opcode_value(value) for value in blocks.inverse()],
        "applied": edits.apply(left, right),
        "opcodeApplied": blocks.apply(left, right),
    }


metric_modules = {
    "levenshtein": Levenshtein,
    "indel": Indel,
    "lcs": LCSseq,
    "osa": OSA,
    "damerauLevenshtein": DamerauLevenshtein,
    "hamming": Hamming,
    "jaro": Jaro,
    "jaroWinkler": JaroWinkler,
    "prefix": Prefix,
    "postfix": Postfix,
}

metric_cases = []
for family, module in metric_modules.items():
    for left, right in BASE_CASES:
        metric_cases.append(metric_case(family, module, left, right))

metric_cases.extend(
    [
        metric_case(
            "levenshtein",
            Levenshtein,
            "lewenstein",
            "levenshtein",
            {"weights": [1, 1, 2]},
            {"weights": (1, 1, 2)},
        ),
        metric_case(
            "hamming",
            Hamming,
            "karolin",
            "kathrin",
            {"pad": False},
            {"pad": False},
        ),
        metric_case(
            "jaroWinkler",
            JaroWinkler,
            "martha",
            "marhta",
            {"prefixWeight": 0.2},
            {"prefix_weight": 0.2},
        ),
    ]
)

fuzz_scorers = {
    "similarity": fuzz.ratio,
    "partialSimilarity": fuzz.partial_ratio,
    "tokenSortSimilarity": fuzz.token_sort_ratio,
    "tokenSetSimilarity": fuzz.token_set_ratio,
    "tokenSimilarity": fuzz.token_ratio,
    "partialTokenSortSimilarity": fuzz.partial_token_sort_ratio,
    "partialTokenSetSimilarity": fuzz.partial_token_set_ratio,
    "partialTokenSimilarity": fuzz.partial_token_ratio,
    "weightedSimilarity": fuzz.WRatio,
}
fuzz_inputs = [
    ["", ""],
    ["new york mets", "new YORK mets"],
    ["fuzzy was a bear", "fuzzy fuzzy was a bear"],
    ["alpha beta", "beta alpha"],
    ["😀 alpha", "alpha 😀"],
]
fuzz_cases = [
    {
        "left": left,
        "right": right,
        "scores": {name: scorer(left, right) for name, scorer in fuzz_scorers.items()},
    }
    for left, right in fuzz_inputs
]

editop_modules = {
    "levenshtein": Levenshtein,
    "indel": Indel,
    "lcs": LCSseq,
    "hamming": Hamming,
}

EDITOP_PAIRS = [
    ["qabxcd", "abycdf"],
    ["😀ab", "😀ac"],
    ["", ""],
    ["abc", "abc"],
    ["", "abc"],
    ["abc", ""],
    ["aaa", "aaaa"],
    ["ΣΊΣΥΦΟΣ", "σίσυφος"],
]

editops_cases = [
    editops_detail(family, module, left, right)
    for family, module in editop_modules.items()
    for left, right in EDITOP_PAIRS
]

# The only editops taking a third argument. `pad=False` refuses unequal lengths,
# and that refusal is part of the contract.
hamming_editops_cases = [
    editops_detail("hamming", Hamming, left, right, {"pad": False}, {"pad": False})
    for left, right in [["karolin", "kathrin"], ["abc", "abc"], ["", ""]]
]
hamming_editops_errors = [
    {"left": "ab", "right": "abc", "configuration": {"pad": False}, "error": message}
    for message in ["Sequences are not the same length."]
]

# `remove_subsequence` requires its argument to be a genuine subsequence of the
# receiver — the C++ path segfaults otherwise, so the triple is fixed, not generated.
# This one carries an insert and a delete, which is what exercises the offset
# arithmetic; an all-`replace` triple leaves it at zero.
remove_subsequence_cases = []
for source, subset, target in [["abcd", "baac", "baaa"], ["abcd", "aaaa", "bxaa"]]:
    full = Levenshtein.editops(source, target)
    part = Levenshtein.editops(source, subset)
    remainder = full.remove_subsequence(part)
    remove_subsequence_cases.append(
        {
            "family": "levenshtein",
            "source": source,
            "subset": subset,
            "target": target,
            "full": [editop_value(value) for value in full],
            "subsequence": [editop_value(value) for value in part],
            "operations": [editop_value(value) for value in remainder],
            "srcLen": remainder.src_len,
            "destLen": remainder.dest_len,
        }
    )

alignment_cases = [
    {"left": left, "right": right, "alignment": alignment_value(alignment)}
    for left, right, alignment in (
        (left, right, fuzz.partial_ratio_alignment(left, right))
        for left, right in [
            ["", ""],
            ["a", ""],
            ["abc", "xyz"],
            ["fuzzy was a bear", "fuzzy fuzzy was a bear"],
            ["new york mets", "the new york mets played"],
            ["😀 alpha", "alpha 😀"],
        ]
    )
]

search_query = "new york mets"
search_choices = [
    "new york mets",
    "new york jets",
    "the mets from new york",
    "atlanta braves",
    "new york mets",
]
search_threshold = 60

batch_queries = ["kitten", "abc", "😀a"]
batch_choices = ["sitting", "axc", "😀b"]
matrix_choices = ["sitting", "kitten", "axc", "😀b"]

# `abcdefgh`/`abcdefgx` is 7/8, so scaling it lands on 87.5 — the tie that shows
# which way an integral kind rounds.
dtype_queries = ["abcdefgh", "kitten", "abc"]
dtype_choices = ["abcdefgx", "sitting", "abc"]
NUMPY_KINDS = {
    "f64": np.float64,
    "f32": np.float32,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "u32": np.uint32,
    "u16": np.uint16,
    "u8": np.uint8,
}


def batch_kind(kind, dtype):
    integral = np.dtype(dtype).kind in "iu"
    multiplier = 100 if integral else 1
    return {
        "into": kind,
        "scoreMultiplier": multiplier,
        "pairs": process.cpdist(
            dtype_queries,
            dtype_choices,
            scorer=Levenshtein.normalized_similarity,
            score_multiplier=multiplier,
            dtype=dtype,
        ).tolist(),
    }


batch_dtype_cases = [batch_kind(kind, dtype) for kind, dtype in NUMPY_KINDS.items()]

# `u8c` clamps where `u8` wraps, and numpy has no clamped analogue. Every score
# here is within 0..255, where the two agree — clamping itself is ours to test.
batch_dtype_cases.append(dict(batch_kind("u8c", np.uint8), into="u8c"))

batch_rejection_cases = [
    {
        "metric": name,
        "threshold": threshold,
        "pairs": process.cpdist(
            dtype_queries,
            dtype_choices,
            scorer=scorer,
            score_cutoff=threshold,
            dtype=np.float64,
        ).tolist(),
    }
    for name, scorer, threshold in [
        ("levenshtein.distance", Levenshtein.distance, 2),
        ("levenshtein.distance", Levenshtein.distance, 0),
        ("levenshtein.normalizedSimilarity", Levenshtein.normalized_similarity, 0.9),
        ("fuzz.similarity", fuzz.ratio, 90),
    ]
]

# A record and a Map both derive a key the way upstream's dict does.
record_choices = {
    "mets": "new york mets",
    "jets": "new york jets",
    "braves": "atlanta braves",
}

normalized_choices = ["NEW YORK METS!!!", "new_york_jets", "  Atlanta Braves  "]
normalized_query = "New YORK Mets"


# Every case where the C++ extension and the pure-Python fallback disagree, with
# the side this port follows. An entry without `follow` is a divergence nobody has
# classified yet, and the parity test refuses it.
DEFECTS = {
    # One backend answers two different things for inputs that mean the same.
    # Whichever side is self-consistent is the correct one, and its own other
    # spellings are the oracle that says so.
    "self-contradiction",
    # A value the metric's own domain cannot contain.
    "out-of-domain-value",
    # An internal implementation detail escaping as an exception.
    "leaked-internal-error",
    # Both backends are defective, and the consistent answer is a third one.
    "both-defective",
}


def divergence(id, surface, cpp, py, correct, defect, reason, evidence=(), **extra):
    if cpp == py:
        raise SystemExit(f"divergence {id} no longer diverges: both backends say {cpp!r}")
    if correct not in {"cpp", "py", "ours"}:
        raise SystemExit(f"divergence {id} has no verdict")
    if defect not in DEFECTS:
        raise SystemExit(f"divergence {id} names an unknown defect {defect!r}")
    return dict(
        {
            "id": id,
            "surface": surface,
            "cpp": cpp,
            "py": py,
            "correct": correct,
            "defect": defect,
            "reason": reason,
            "evidence": list(evidence),
        },
        **extra,
    )


def token_pair(separator):
    return f"b{separator}a zz", f"a{separator}b zz"


def token_evidence(note, left, right):
    return {
        "note": note,
        "left": left,
        "right": right,
        "cpp": fuzz.token_sort_ratio(left, right),
        "py": fuzz_py.token_sort_ratio(left, right),
    }


def token_divergence(id, separator, name):
    left, right = token_pair(separator)
    # Each item is a spelling of the same comparison that C++ *does* split, so the
    # verdict rests on C++ disagreeing with itself rather than on a preference.
    wide_left, wide_right = token_pair(separator)
    evidence = [
        token_evidence(
            "the same text with one token widened past latin-1",
            wide_left + " 一",
            wide_right + " 一",
        ),
        token_evidence(
            "the same text as code points, which is what a list input is",
            [ord(c) for c in left],
            [ord(c) for c in right],
        ),
        token_evidence(
            "U+202F, also a non-breaking space and also Zs, but too wide for latin-1",
            *token_pair(chr(0x202F)),
        ),
    ]
    return divergence(
        id,
        "fuzz.tokenSortSimilarity",
        fuzz.token_sort_ratio(left, right),
        fuzz_py.token_sort_ratio(left, right),
        "py",
        "self-contradiction",
        f"{name} is str.isspace() and Zs/Cc, and `_py` splits on it everywhere. C++ "
        "splits on it in a wide string and in its list path but not in a latin-1 "
        "string, and splits on U+202F — equally non-breaking — in every case. The "
        "trigger is CPython's storage width, not the character's meaning, so there is "
        "no reading on which the latin-1 answer is the deliberate one.",
        evidence,
        left=left,
        right=right,
    )


jaro_left = "a" * 40
jaro_right = "a" * 39 + "b"

# Truncating a raw cutoff cannot change accept/reject for an integer-valued metric:
# no integer lies strictly between trunc(c) and c. It only changes the rejection
# sentinel — and C++ truncates its *weights* too, so its distance is always integral.
weight_evidence = [
    {
        "note": "C++ truncates fractional weights, so a raw distance is always integral",
        "weights": list(weights),
        "cpp": Levenshtein.distance("kitten", "sitting", weights=weights),
    }
    for weights in [(1, 1, 2), (1, 1, 1.5), (0.5, 0.5, 1)]
]

divergences = [
    token_divergence("token-split-u0085", chr(0x0085), "U+0085 NEL"),
    token_divergence("token-split-u00a0", chr(0x00A0), "U+00A0 NBSP"),
    divergence(
        "jaro-winkler-bounded-ulp",
        "jaroWinkler.similarity",
        JaroWinkler.similarity(jaro_left, jaro_right, score_cutoff=0.99),
        JaroWinkler_py.similarity(jaro_left, jaro_right, score_cutoff=0.99),
        "py",
        "self-contradiction",
        "score_cutoff promises the score back when score >= cutoff. The score is "
        "exactly 0.99 and the cutoff is exactly 0.99, so it has to be returned. C++ "
        "returns 0.99 unbounded and 0.0 bounded by that same value, because its "
        "bounded Jaro bails one ULP early.",
        [
            {
                "note": "C++ contradicting itself: the same pair, no cutoff",
                "cpp": JaroWinkler.similarity(jaro_left, jaro_right),
                "py": JaroWinkler_py.similarity(jaro_left, jaro_right),
            }
        ],
        left=jaro_left,
        right=jaro_right,
        threshold=0.99,
    ),
    divergence(
        "fractional-raw-threshold",
        "scorePairs(levenshtein.distance)",
        process.cpdist(
            ["kitten"], ["sitting"], scorer=Levenshtein.distance,
            score_cutoff=1.9, dtype=np.float64,
        ).tolist(),
        process_py.cpdist(
            ["kitten"], ["sitting"], scorer=Levenshtein_py.distance,
            score_cutoff=1.9, dtype=np.float64,
        ).tolist(),
        "cpp",
        "out-of-domain-value",
        "A rejected pair is stored as a value that beats the cutoff. `_py` stores 2.9, "
        "which no raw edit distance can ever be — upstream truncates its weights, so "
        "the metric is integer-valued — and which no integral `into` kind can hold. "
        "C++'s 2 is in-domain and unambiguous, since a real distance of 2 is itself "
        "rejected at a cutoff of 1.9. Accept/reject is identical either way.",
        weight_evidence,
        queries=["kitten"],
        choices=["sitting"],
        threshold=1.9,
    ),
]

# A whitespace element of more than one character. The earlier version of this case
# compared a sequence with itself, which is 100.0 whether or not the element splits —
# it could not tell the two behaviours apart. Swapping the outer tokens can.
multichar_left = ["b", "  ", "a"]
multichar_right = ["a", "  ", "b"]
try:
    multichar_py = fuzz_py.token_sort_ratio(multichar_left, multichar_right)
    multichar_error = None
except Exception as error:  # noqa: BLE001 - the refusal is the finding
    multichar_py = None
    multichar_error = f"{type(error).__name__}: {error}"

divergences.append(
    divergence(
        "multi-character-element",
        "fuzz.tokenSortSimilarity",
        fuzz.token_sort_ratio(multichar_left, multichar_right),
        multichar_py,
        "ours",
        "both-defective",
        "`_py` raises out of chr() — an internal detail, not a decision. C++ answers, "
        "but splits this same whitespace when it is one element and when it is a code "
        "point, and declines only when it is two characters. Both agree on the "
        "equivalent spellings below, and that shared answer is the consistent one.",
        [
            {
                "note": "the same comparison with a single-character element",
                "left": ["b", " ", "a"],
                "right": ["a", " ", "b"],
                "cpp": fuzz.token_sort_ratio(["b", " ", "a"], ["a", " ", "b"]),
                "py": fuzz_py.token_sort_ratio(["b", " ", "a"], ["a", " ", "b"]),
            },
            {
                "note": "the same comparison as code points",
                "left": [98, 32, 97],
                "right": [97, 32, 98],
                "cpp": fuzz.token_sort_ratio([98, 32, 97], [97, 32, 98]),
                "py": fuzz_py.token_sort_ratio([98, 32, 97], [97, 32, 98]),
            },
        ],
        left=multichar_left,
        right=multichar_right,
        pyError=multichar_error,
    )
)

fixture = {
    "rapidfuzzVersion": rapidfuzz.__version__,
    "metricCases": metric_cases,
    "fuzzCases": fuzz_cases,
    "normalization": [
        {"input": value, "output": utils.default_process(value)}
        for value in [
            "  New YORK Mets!!!  ",
            "Hello 😀 World! 👍🏽",
            "Straße—Zürich",
            "Already_normalized",
            "!!!",
            "",
            "İstanbul",
            "DİYARBAKIR İZMİR",
            "ΟΔΟΣ ΟΔΟΣ",
            "ΣΊΣΥΦΟΣ",
            "ΑΣ",
            "ΑΣΑ",
            "ΣΣΣ",
            "Grüße" + chr(0x00A0) + "Welt",
            "line" + chr(0x0085) + "break",
            "tab\tand\nnewline",
            "a---b",
            "  ",
        ]
    ],
    "editopsCases": editops_cases,
    "hammingEditopsCases": hamming_editops_cases,
    "hammingEditopsErrors": hamming_editops_errors,
    "removeSubsequenceCases": remove_subsequence_cases,
    "alignmentCases": alignment_cases,
    "divergences": divergences,
    "search": {
        "query": search_query,
        "choices": search_choices,
        "threshold": search_threshold,
        "iter": [
            {"item": item, "score": score, "key": key}
            for item, score, key in process.extract_iter(
                search_query,
                search_choices,
                scorer=fuzz.ratio,
                score_cutoff=search_threshold,
            )
        ],
        "best": dict(
            zip(
                ["item", "score", "key"],
                process.extractOne(search_query, search_choices, scorer=fuzz.ratio),
            )
        ),
        "top": [
            {"item": item, "score": score, "key": key}
            for item, score, key in process.extract(
                search_query,
                search_choices,
                scorer=fuzz.ratio,
                score_cutoff=search_threshold,
                limit=4,
            )
        ],
        "record": {
            "choices": record_choices,
            "top": [
                {"item": item, "score": score, "key": key}
                for item, score, key in process.extract(
                    search_query,
                    record_choices,
                    scorer=fuzz.ratio,
                    score_cutoff=search_threshold,
                    limit=None,
                )
            ],
        },
        "normalized": {
            "query": normalized_query,
            "choices": normalized_choices,
            "threshold": search_threshold,
            "best": dict(
                zip(
                    ["item", "score", "key"],
                    process.extractOne(
                        normalized_query,
                        normalized_choices,
                        scorer=fuzz.ratio,
                        processor=utils.default_process,
                    ),
                )
            ),
            "top": [
                {"item": item, "score": score, "key": key}
                for item, score, key in process.extract(
                    normalized_query,
                    normalized_choices,
                    scorer=fuzz.ratio,
                    processor=utils.default_process,
                    score_cutoff=search_threshold,
                    limit=None,
                )
            ],
        },
    },
    "batchDtypes": {
        "queries": dtype_queries,
        "choices": dtype_choices,
        "kinds": batch_dtype_cases,
    },
    "batchRejections": {
        "queries": dtype_queries,
        "choices": dtype_choices,
        "cases": batch_rejection_cases,
    },
    "batch": {
        "queries": batch_queries,
        "choices": batch_choices,
        "matrixChoices": matrix_choices,
        "threshold": 0.5,
        "scoreMultiplier": 100,
        "pairs": process.cpdist(
            batch_queries,
            batch_choices,
            scorer=Levenshtein.normalized_similarity,
            score_cutoff=0.5,
            score_multiplier=100,
            dtype=np.uint8,
        ).tolist(),
        "matrix": process.cdist(
            batch_queries,
            matrix_choices,
            scorer=Levenshtein.normalized_similarity,
            score_cutoff=0.5,
            score_multiplier=100,
            dtype=np.uint8,
        ).tolist(),
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
print(OUTPUT)
