---
title: Performance
description: The habits that matter, in order — Matcher reuse, thresholds, typed arrays, bundle size.
---

Fuzzy matching is CPU work, and the difference between a fast and a slow
integration is rarely the algorithm — it's how you call it. In order of
impact:

## 1. Reuse a Matcher

Before scoring, every algorithm builds internal structures from the text.
One-shot calls rebuild them for the whole collection on every query; a
[Matcher](/concepts/matchers/) builds them once at construction.

| Workload                      | Use                               |
| ----------------------------- | --------------------------------- |
| One best result, once         | `bestMatch`                       |
| Top N results, once           | `search` with `limit: N`          |
| The same collection, many queries | `createMatcher`, then `matcher.best` / `matcher.search` |

The [benchmarks](/benchmarks/) put numbers on it: prepared-input paths
measured up to **7×** faster than re-preparing per call on token workloads.
If you remember one thing from this page: *can you name a second query?
Build a Matcher.*

When a Matcher can't own the collection — you filter it per query, grow it,
or keep it in your own index — the same preparation is available piecemeal:
`scorer.prepareChoice` each candidate once and pass `getPrepared` to the
one-shot searches. Same speed class, collection stays yours —
[Prepared choices](/guides/prepared-choices/).

## 2. Set real thresholds

A `threshold` isn't just a result filter — the edit-distance algorithms use
it as a cutoff and **abandon pairs mid-computation** once they can't
qualify. During a `best` search the current front-runner tightens the cutoff
further as it goes, so obviously-wrong candidates cost almost nothing.

Filtering results yourself afterwards gets the same answers while paying
full price for every reject. Put the bar in the call.

## 3. Score in bulk with typed arrays

`scoreMatrix` and `scorePairs` write all scores into one typed array — no
per-score object allocation, no GC pressure in a hot loop. With `into` you
choose the element size; `u8` fits any 0–100 fuzz score in an eighth of
`f64`'s memory, and typed-array buffers transfer to workers and WebAssembly
without copying. See
[Comparing strings](/guides/comparing-strings/#many-pairs-at-once).

## 4. Let the bundler drop the rest

Every algorithm lives on its own subpath, and the package declares
`sideEffects: false` with nothing executing at import time. Import
`rapidfuzz-js/levenshtein` and your bundle contains Levenshtein — not the
fuzz token machinery, not Jaro, nothing you didn't ask for.

## What you don't need to do

No tuning flags, no warm-up, no worker setup. The kernels are bit-parallel
where it pays and threshold-pruned where it helps — that work is inside the
library, [measured against the ecosystem](/benchmarks/), and not your
problem. The four habits above are the whole checklist.

---

*For the curious: the release pipeline also enforces type checking, 100%
test coverage, export-map validation, and tarball inspection — performance
claims are only trustworthy when the package is exactly what you think it
is.*
