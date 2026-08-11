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

:::note
These results were recorded against the 0.5.x API. The workload names map
onto the current API as described in
[Performance](/guides/performance/#1-reuse-a-matcher); the underlying
algorithm kernels are the same.
:::

## Levenshtein distance

Pairwise distance against dedicated Levenshtein libraries. Results read as
rapidfuzz-js relative to the named library:

| Input                      | `fastest-levenshtein` | `leven`              | `js-levenshtein`    | `fuzzball`           |
| -------------------------- | --------------------- | -------------------- | ------------------- | -------------------- |
| 8 characters, 200 pairs    | ❌ 1.06× slower       | ✅ **1.42× faster**  | ❌ 1.70× slower     | ✅ **2.97× faster**  |
| 32 characters, 200 pairs   | ✅ **1.09× faster**   | ✅ **3.10× faster**  | ✅ **1.94× faster** | ✅ **11.69× faster** |
| 128 characters, 200 pairs  | ✅ **1.36× faster**   | ✅ **9.56× faster**  | ✅ **5.22× faster** | ✅ **15.53× faster** |
| 1,024 characters, 25 pairs | ✅ **1.62× faster**   | ✅ **18.66× faster** | ✅ **9.84× faster** | ✅ **19.83× faster** |

## Similarity and best-match search

| Workload                             | Compared with                   | Result for `rapidfuzz-js` |
| ------------------------------------ | ------------------------------- | ------------------------- |
| Ratio, 200 sentence pairs            | `fuzzball`                      | ✅ **16.35× faster**      |
| Ratio, 200 sentence pairs            | `string-similarity`             | ✅ **27.43× faster**      |
| Best of 2,000 choices for 20 queries | `fuzzball`                      | ✅ **5.34× faster**       |
| Best of 2,000 choices for 20 queries | `string-similarity`             | ✅ **21.87× faster**      |
| Best of 2,000 choices for 20 queries | `fuse.js` with a prebuilt index | ✅ **62.61× faster**      |

## Why prepared inputs matter

Most of the cost of a fuzzy query is preparing the inputs, not scoring them.
Preparing the collection once — which is what
[`createMatcher`](/concepts/matchers/) construction does — removes that cost
from every subsequent query. Measured on the 0.5.x prepared-input API, both
halves prepared reached **7.22× faster** on a token-sort workload of
20 queries × 200 choices; none of the compared JavaScript libraries expose a
prepared-operand mechanism at all.

## Reproducing

The comparison suite lives in
[`bench/comparison/`](https://github.com/sarunast/rapidfuzz-js/tree/main/bench/comparison)
as a self-contained package, pinned and runnable on your own hardware.
