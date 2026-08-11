"""Generate the RapidFuzz 3.14.5 oracle consumed by parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rapidfuzz
from rapidfuzz import fuzz, process, utils
from rapidfuzz.distance import (
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    OSA,
    Postfix,
    Prefix,
)


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "fixtures" / "rapidfuzz-3.14.5.json"

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
    "fuzzySimilarity": fuzz.WRatio,
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

editops_cases = []
for family, module in {
    "levenshtein": Levenshtein,
    "indel": Indel,
    "lcs": LCSseq,
    "hamming": Hamming,
}.items():
    for left, right in [["qabxcd", "abycdf"], ["😀ab", "😀ac"]]:
        editops_cases.append(
            {
                "family": family,
                "left": left,
                "right": right,
                "editops": [editop_value(value) for value in module.editops(left, right)],
                "opcodes": [opcode_value(value) for value in module.opcodes(left, right)],
            }
        )

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
        ]
    ],
    "editopsCases": editops_cases,
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
