---
title: Benchmarks
description: How rapidfuzz-js performs against other JavaScript fuzzy-matching libraries.
---

Fuzzy matching is CPU-bound work, so speed decides what's feasible — a
search-as-you-type box over 10,000 products lives or dies on how fast each
keystroke's query runs. This page shows how rapidfuzz-js compares to the
other JavaScript options. Each cell reads as "rapidfuzz-js was N× faster
(or slower) than the named library on that workload".

Numbers come from the repository's reproducible comparison suite; the full
methodology, environment, and result set live in
[BENCHMARKS.md](https://github.com/sarunast/rapidfuzz-js/blob/main/BENCHMARKS.md).

Recorded in August 2026 on an Apple M1 Max under Node.js 26.5.1, against the
dependency versions pinned in the suite. `rapidfuzz-js` was measured through
its `dist/` build — the same code consumers run.

## Levenshtein distance

Pairwise distance against dedicated Levenshtein libraries. The runner checks
that every contender returns the same distance for every pair before timing
it. Results read as rapidfuzz-js relative to the named library:

| Input                      | `fastest-levenshtein` | `leven`              | `js-levenshtein`    | `fuzzball`           |
| -------------------------- | --------------------- | -------------------- | ------------------- | -------------------- |
| 8 characters, 200 pairs    | ❌ 1.07× slower       | ✅ **1.41× faster**  | ❌ 1.72× slower     | ✅ **2.89× faster**  |
| 32 characters, 200 pairs   | ✅ **1.09× faster**   | ✅ **2.99× faster**  | ✅ **1.91× faster** | ✅ **11.65× faster** |
| 128 characters, 200 pairs  | ✅ **1.34× faster**   | ✅ **9.64× faster**  | ✅ **5.35× faster** | ✅ **15.52× faster** |
| 1,024 characters, 25 pairs | ✅ **1.59× faster**   | ✅ **18.40× faster** | ✅ **9.76× faster** | ✅ **19.50× faster** |

The two dedicated libraries win on eight-character inputs. rapidfuzz-js moves
ahead at 32 characters and widens the lead as inputs grow.

## Similarity and best-match search

| Workload                             | Compared with                   | Result for `rapidfuzz-js` |
| ------------------------------------ | ------------------------------- | ------------------------- |
| `similarity`, 200 sentence pairs     | `fuzzball`                      | ✅ **16.11× faster**      |
| `similarity`, 200 sentence pairs     | `string-similarity`             | ✅ **27.99× faster**      |
| Best of 2,000 choices for 20 queries | `fuzzball`                      | ✅ **5.29× faster**       |
| Best of 2,000 choices for 20 queries | `string-similarity`             | ✅ **22.27× faster**      |
| Best of 2,000 choices for 20 queries | `fuse.js` with a prebuilt index | ✅ **61.61× faster**      |

These rows don't all run the same algorithm. `fuzzball` is the like-for-like
comparison, with preprocessing disabled on both sides and its result checked
against ours. `string-similarity` uses Dice similarity over bigrams;
`fuse.js` uses Bitap with its index built outside the timed loop while
rapidfuzz-js scans every choice. Read the last two rows as runtime for this
corpus, not as a verdict on scoring quality.

## Why a Matcher pays

Most of the cost of a fuzzy query is preparing the inputs, not scoring them,
and [`createMatcher`](/concepts/matchers/) moves that preparation to
construction. Pairing each one-shot search with the `Matcher` search that
replaces it, over the same queries, choices and scorer:

| Workload                                         | One-shot | `Matcher` | Result              |
| ------------------------------------------------ | -------: | --------: | ------------------- |
| 30 queries × 2,000 choices, `similarity`         |  5.09 ms |   3.53 ms | ✅ **1.44× faster** |
| 30 queries × 2,000 titles, `tokenSortSimilarity` |  64.4 ms |   9.71 ms | ✅ **6.63× faster** |

Construction is what buys that, and it is paid once — 0.058 ms for the 2,000
single-word choices, 0.939 ms for the 2,000 normalized five-word titles. The
token-sort case gains most because tokenizing and sorting 2,000 titles
happens per query in the one-shot loop and once for the `Matcher`; plain
`similarity` gains least because its reusable setup is a smaller share of
the work.

None of the compared JavaScript libraries expose a prepared-operand
mechanism; `fuse.js` reuses a collection index, but over a different
matching algorithm.

## Where rapidfuzz-js loses

Python RapidFuzz is faster on most substantial workloads, especially ones
that complete entirely inside its C++ extension — worth knowing if you're
choosing between runtimes rather than between JavaScript libraries.

## Reproducing

The comparison suite lives in
[`bench/comparison/`](https://github.com/sarunast/rapidfuzz-js/tree/main/bench/comparison)
as a self-contained package, pinned and runnable on your own hardware.
