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

## Dice and Cosine

Dice and Cosine were added after the pass that produced the tables above, so
these rows come from a later one — every figure here is a ratio between two
libraries timed in the same process, reproduced across three runs. The runner
checks that every contender in this table returns our number for every pair in
the corpus before timing it: all four take Dice over a _multiset_ of bigrams,
which is what `similarity` from `rapidfuzz-js/dice` does.

| Input                      | `dice-coefficient` | `fast-dice-coefficient` | `string-similarity` | `string-comparison` |
| -------------------------- | ------------------ | ----------------------- | ------------------- | ------------------- |
| 8 characters, 200 pairs    | ❌ 2.51× slower    | ✅ **1.24× faster**     | ✅ **1.26× faster** | ✅ **1.29× faster** |
| 32 characters, 200 pairs   | ❌ 1.74× slower    | ✅ **1.22× faster**     | ✅ **1.28× faster** | ✅ **1.40× faster** |
| 128 characters, 200 pairs  | ✅ **1.44×**       | ✅ **1.30× faster**     | ✅ **1.36× faster** | ✅ **1.35× faster** |
| 1,024 characters, 25 pairs | ✅ **10.00×**      | ✅ **1.42× faster**     | ✅ **1.45× faster** | ✅ **1.39× faster** |

`dice-coefficient` is the one library that beats us, and it is worth
understanding why. It compares two bigram arrays with a nested scan — no map,
no allocation — which is `O(n·m)` but has nothing to build. Below roughly 64
characters that trade wins; above it the quadratic term takes over, and by
1,024 characters it is an order of magnitude behind. If your inputs are short
words and Dice is all you need, it is the faster choice.

Cosine has almost no competition: `wink-nlp-utils` bagging n-grams with
`wink-distance` taking the cosine of the two bags is the only other true
n-gram cosine in JavaScript, and it agrees with ours to 2e-16.

| Input                      | `wink` bag-of-n-grams + `bow.cosine` |
| -------------------------- | ------------------------------------ |
| 8 characters, 200 pairs    | ✅ **3.16× faster**                  |
| 32 characters, 200 pairs   | ✅ **3.19× faster**                  |
| 128 characters, 200 pairs  | ✅ **3.39× faster**                  |
| 1,024 characters, 25 pairs | ✅ **2.57× faster**                  |

At `gramSize: 3` over 128 characters, Dice is 1.20× faster than
`dice-coefficient` and Cosine 3.35× faster than the wink pair.

Searching is where the gap widens, because a `Matcher` profiles each choice
once instead of once per query — 20 queries over 2,000 choices:

| Workload          | Compared with                          | Result for `rapidfuzz-js` |
| ----------------- | -------------------------------------- | ------------------------- |
| `bestMatch`, Dice | `string-similarity.findBestMatch`      | ✅ **1.68× faster**       |
| `bestMatch`, Dice | `fast-dice-coefficient` loop           | ✅ **1.58× faster**       |
| `bestMatch`, Dice | `dice-coefficient` loop                | ❌ 1.10× slower           |
| `Matcher`, Dice   | `string-similarity.findBestMatch`      | ✅ **6.41× faster**       |
| `Matcher`, Dice   | `fast-dice-coefficient` loop           | ✅ **6.28× faster**       |
| `Matcher`, Dice   | `dice-coefficient` with prebuilt grams | ✅ **4.17× faster**       |
| `Matcher`, Cosine | `wink` loop                            | ✅ **20.04× faster**      |
| `Matcher`, Cosine | `wink` with prebuilt bags              | ✅ **12.62× faster**      |

Three contenders are deliberately absent from the tables above, because they
do not compute the same thing:

- `talisman/metrics/dice` and `natural.DiceCoefficient` take Dice over a
  **set** of bigrams, so `'aaaa'` against `'aaa'` is `1` where we answer
  `0.8`. We measure 1.45–2.79× faster than both, but they are doing less work.
- `string-comparison`'s `cosine` is a binary vector over **characters**, not a
  frequency vector over n-grams — it scores `'iwmaxzsz'` against `'iwmaxssz'`
  a flat `1`. It is 1.40–2.82× faster than our Cosine and answers a different
  question.

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
