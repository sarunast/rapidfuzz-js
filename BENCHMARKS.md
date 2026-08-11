# Benchmarks

This document explains how `rapidfuzz-js` performs, how the measurements are
calculated, and how to reproduce them.

Every number below comes from a single recorded pass of
[`bench/comparison/run.mjs`](bench/comparison/run.mjs), plus one pass of the
internal `bench/process.bench.ts` for the `Matcher` table. Rerunning replaces
all of them together; do not update one table from a later run.

## Summary

The recorded comparison shows that `rapidfuzz-js`:

- is close to the fastest specialized JavaScript Levenshtein libraries on
  short strings and faster on longer strings;
- is 2.9–19.5× faster than `fuzzball` for the measured Levenshtein workloads;
- is about 16× faster than `fuzzball` for `similarity` and 5× faster for
  best-match search;
- answers repeated queries over a stable collection 5.14× faster through a
  `Matcher` than through the same one-shot search, and scores a held query
  1.29× faster than scoring each pair directly;
- makes `Matcher` token-sort search 3.40× faster than Python RapidFuzz and
  18.83× faster than `fuzzball` on the measured stable-catalog workload; and
- is slower than Python RapidFuzz on most substantial workloads, especially
  operations completed entirely inside its C++ extension.

These results describe the workloads and environment below. They are not a
guarantee for every input, runtime, or machine.

## How to read the results

All comparison tables show the result for `rapidfuzz-js` directly:

- ✅ **Faster** means `rapidfuzz-js` won by the displayed factor.
- ❌ Slower means the competing library won by the displayed factor.
- ≈ Same means the measured difference was below 1%.

## Recorded environment

The results below were recorded in August 2026 with:

- Apple M1 Max, macOS arm64
- Node.js 26.5.1
- Python 3.14.6, RapidFuzz 3.14.5, and NumPy 2.5.2 for the optional Python
  comparison
- the dependency versions pinned in
  [`bench/comparison/package.json`](bench/comparison/package.json)

Every JavaScript contender used its installed package build. `rapidfuzz-js`
used `dist/`, matching the code consumers run.

## JavaScript comparisons

### Levenshtein distance

The runner verifies that every contender returns the same distance for every
pair before measuring it.

Each cell describes `rapidfuzz-js` relative to the library in that column.

| Input                      | `fastest-levenshtein` | `leven`              | `js-levenshtein`    | `fuzzball`           |
| -------------------------- | --------------------- | -------------------- | ------------------- | -------------------- |
| 8 characters, 200 pairs    | ❌ 1.07× slower       | ✅ **1.41× faster**  | ❌ 1.72× slower     | ✅ **2.89× faster**  |
| 32 characters, 200 pairs   | ✅ **1.09× faster**   | ✅ **2.99× faster**  | ✅ **1.91× faster** | ✅ **11.65× faster** |
| 128 characters, 200 pairs  | ✅ **1.34× faster**   | ✅ **9.64× faster**  | ✅ **5.35× faster** | ✅ **15.52× faster** |
| 1,024 characters, 25 pairs | ✅ **1.59× faster**   | ✅ **18.40× faster** | ✅ **9.76× faster** | ✅ **19.50× faster** |

For eight-character inputs, `fastest-levenshtein` and `js-levenshtein` were
faster. `rapidfuzz-js` moved ahead of `fastest-levenshtein` at 32 characters
and widened the lead as input length increased.

### Similarity and best-match search

| Workload                             | Compared with                   | Result for `rapidfuzz-js` |
| ------------------------------------ | ------------------------------- | ------------------------- |
| `similarity`, 200 sentence pairs     | `fuzzball`                      | ✅ **16.11× faster**      |
| `similarity`, 200 sentence pairs     | `string-similarity`             | ✅ **27.99× faster**      |
| Best of 2,000 choices for 20 queries | `fuzzball`                      | ✅ **5.29× faster**       |
| Best of 2,000 choices for 20 queries | `string-similarity`             | ✅ **22.27× faster**      |
| Best of 2,000 choices for 20 queries | `fuse.js` with a prebuilt index | ✅ **61.61× faster**      |

The comparisons do not all use the same algorithm:

- `fuzzball` is the closest like-for-like comparison. Preprocessing is disabled
  on both sides, and its rounded `ratio` result is checked against ours.
- `string-similarity` uses Dice similarity over bigrams rather than
  Indel-normalized similarity.
- `fuse.js` uses Bitap with an index built outside the timed loop.
  `rapidfuzz-js` scans every choice in this test.

The last two rows compare complete matching tasks, not identical scoring
algorithms. Use them to understand runtime for this corpus, not scoring quality
or semantic equivalence.

## Held inputs: what a Matcher reuses

Most scorers do more than calculate the final score. Depending on the scorer
and normalizer, each call may normalize text, split and sort tokens, deduplicate
tokens, build character masks, or create another query representation. When an
operand is reused, repeating that setup produces the same intermediate data
again.

`rapidfuzz-js` holds that reusable work in three places, and none of them is a
handle the caller has to manage:

- A one-shot `bestMatch`, `search` or `searchIter` call prepares the query once
  and reuses it across every choice in that call.
- `scoreMatrix` prepares every row once and every column once, so an `R × C`
  matrix costs `R + C` preparations rather than `R × C`.
- A `Matcher` prepares every choice when it is constructed and reuses those
  preparations for every later `best`, `search` and `searchIter` call.

Public prepared-operand handles were removed in 0.6. A collection is held by
constructing a `Matcher`; a query is held for the length of the call that uses
it.

### How much work is avoided

For `Q` queries and `C` choices, the conceptual setup counts are:

| Approach                               | Query setup | Choice setup | Pair scores |
| -------------------------------------- | ----------: | -----------: | ----------: |
| Direct `Metric` pair-scoring loop      |     `Q × C` |      `Q × C` |     `Q × C` |
| One-shot `bestMatch` / `search`        |         `Q` |      `Q × C` |     `Q × C` |
| `scoreMatrix` over the same inputs     |         `Q` |          `C` |     `Q × C` |
| `createMatcher` once, then `Q` queries |         `Q` |          `C` |     `Q × C` |

Preparation does not skip required comparisons. It removes repeated setup
around those comparisons. This is why the benefit grows when a collection is
searched many times and when its scorer performs expensive token or mask setup.

### Measured effect

The internal suite pairs each one-shot search with the `Matcher` search that
replaces it, over the same queries, choices and scorer:

| Workload                                         | One-shot | `Matcher` | Result              |
| ------------------------------------------------ | -------: | --------: | ------------------- |
| 30 queries × 2,000 choices, `similarity`         |  5.09 ms |   3.53 ms | ✅ **1.44× faster** |
| 30 queries × 2,000 titles, `tokenSortSimilarity` |  64.4 ms |   9.71 ms | ✅ **6.63× faster** |

Construction is what buys that, and it is paid once: 0.058 ms for the 2,000
single-word choices, and 0.939 ms for the 2,000 five-word titles with
normalization. The token-sort case gains most because tokenizing and sorting
2,000 titles is repeated for every query in the one-shot loop and done once for
the `Matcher`. Plain `similarity` gains least because its reusable setup is a
smaller share of the total work.

See [`bench/process.bench.ts`](bench/process.bench.ts) for the paired cases and
the [README](README.md#one-query-or-many) for usage. The suite does not yet
carry a one-shot partner for the normalized-edit `Matcher` case, so that family
has no paired figure here.

### How this differs from the comparison libraries

The pinned library versions in this benchmark expose these reuse options:

| Library               | Public reuse mechanism                  | Practical difference                                                                        |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------- |
| `rapidfuzz-js`        | `Matcher` over a collection             | Prepares every choice once at construction and reuses it for every later query              |
| `fastest-levenshtein` | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                     |
| `leven`               | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                     |
| `js-levenshtein`      | No prepared-operand API                 | Exposes pair scoring only                                                                   |
| `fuzzball`            | No prepared-operand API                 | Scorers and `extract` process inputs within each operation                                  |
| `string-similarity`   | No prepared-operand API                 | Rebuilds bigrams during `compareTwoStrings` calls                                           |
| `fuse.js`             | Prebuilt collection index               | Reuses collection state, but uses Bitap and does not expose prepared scorer operands        |
| Python RapidFuzz      | Internal query preparation in `process` | Reuses a query within one search call, but does not expose a persistent prepared collection |

This does not mean every comparison table should be multiplied by the `Matcher`
speedups. The cross-library Levenshtein cases score independent pairs, where
there is no reusable operand. The similarity and search comparison uses raw
`similarity` inputs with preprocessing disabled, while the largest `Matcher`
gains come from repeated token scoring. Those are different workloads.

Fuse is also not an unindexed comparison: its index is built before the timed
search loop. Its 61.61× loss in the search table already compares a prebuilt
Fuse index with an unindexed `rapidfuzz-js` scan, although the two libraries use
different matching algorithms.

A `Matcher` is most useful for stable data searched or joined repeatedly, such
as autocomplete catalogs, deduplication batches, record linkage, and many-query
ranking. For a one-off search, construction adds setup that may not be
recovered, and a `Matcher` has to be rebuilt after its source data changes.

### Holding inputs versus other libraries

Two cross-library workloads isolate how holding an input changes the result.

For a fixed 128-character Levenshtein query scored against 200 choices, where
the query is held for one `scoreMatrix` row rather than rebuilt per pair:

| Comparison            | External time | Held `rapidfuzz-js` | Holding gain | Overall result       |
| --------------------- | ------------: | ------------------: | -----------: | -------------------- |
| `rapidfuzz-js` direct |        409 µs |              318 µs | ✅ **1.29×** | Holding alone        |
| `fastest-levenshtein` |        905 µs |              323 µs |            — | ✅ **2.80× faster**  |
| `leven`               |       9.09 ms |              320 µs |            — | ✅ **28.39× faster** |
| `js-levenshtein`      |       4.45 ms |              316 µs |            — | ✅ **14.06× faster** |
| `fuzzball`            |       9.16 ms |              326 µs |            — | ✅ **28.10× faster** |
| Python RapidFuzz      |        285 µs |              332 µs |            — | ❌ 1.16× slower      |

Holding the query makes the JavaScript path 1.29× faster than its own direct
call. It also narrows the Python gap from 1.44× slower to 1.16× slower. The
much larger leads over the JavaScript competitors combine this with
`rapidfuzz-js`'s bit-parallel Levenshtein implementation; holding a query alone
does not explain a 28× difference.

Token-sort search shows a larger effect because tokenization and sorting are
reused for the whole collection. The paired JavaScript and `fuzzball` pass
measures it directly:

| 20 queries × 2,000 titles |    Time | Result                            |
| ------------------------- | ------: | --------------------------------- |
| `rapidfuzz-js`, one-shot  | 31.1 ms | Baseline                          |
| `rapidfuzz-js`, `Matcher` | 6.05 ms | ✅ **5.14× faster than one-shot** |
| `fuzzball`, raw titles    |  106 ms | ✅ **Matcher is 18.83× faster**   |

The separately sequenced Python pass compares both JavaScript modes with the
same Python measurement:

| 20 queries × 2,000 titles    |    Time | Result versus Python |
| ---------------------------- | ------: | -------------------- |
| `rapidfuzz-js`, one-shot     | 31.6 ms | ❌ 1.64× slower      |
| `rapidfuzz-js`, `Matcher`    | 5.67 ms | ✅ **3.40× faster**  |
| Python RapidFuzz, raw titles | 19.3 ms | Reference            |

The one-shot JavaScript time moved between the same-process and cross-runtime
passes, which is why the `Matcher` gain is taken from the paired 31.1 ms versus
6.05 ms measurement. This avoids presenting timing drift as a `Matcher`
speedup.

Neither `fuzzball` nor Python RapidFuzz exposes a persistent prepared-choice
collection for this scorer. They prepare a query within a search operation but
process the title collection again for each new query. A `Matcher` pays that
title-side work once and reuses it across the 20 searches.

`string-similarity` and Fuse are not included in this table because they do not
implement the same token-sort scorer. Fuse's prebuilt Bitap index is covered
separately in the best-match table; comparing it here would mix collection
reuse with a different matching algorithm.

## Python RapidFuzz comparison

The Python leg uses the same seeded corpus, loop shapes, warm-up count, pass
length, and median statistic as the JavaScript leg. Times are medians for the
complete workload in each row, not for one string pair.

### Distance and similarity metrics

| Workload                                            | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| --------------------------------------------------- | -------------: | ---------------: | ------------------------- |
| Levenshtein, 200 pairs × 8 characters               |        13.6 µs |          24.0 µs | ✅ **1.76× faster**       |
| Levenshtein, 200 pairs × 32 characters              |        40.0 µs |          33.4 µs | ❌ 1.20× slower           |
| Levenshtein, 200 pairs × 128 characters             |         394 µs |           226 µs | ❌ 1.75× slower           |
| Levenshtein, 25 pairs × 1,024 characters            |        2.27 ms |          1.03 ms | ❌ 2.21× slower           |
| Fixed-query Levenshtein, direct, 1 × 200 × 128      |         412 µs |           285 µs | ❌ 1.44× slower           |
| Fixed-query Levenshtein, held, 1 × 200 × 128        |         332 µs |           285 µs | ❌ 1.16× slower           |
| Indel distance, 200 pairs × 128 characters          |         342 µs |          69.3 µs | ❌ 4.93× slower           |
| LCS similarity, 200 pairs × 128 characters          |         345 µs |          68.9 µs | ❌ 5.00× slower           |
| OSA distance, 200 pairs × 128 characters            |         761 µs |           252 µs | ❌ 3.03× slower           |
| Damerau-Levenshtein, 200 pairs × 128 characters     |        10.9 ms |          5.48 ms | ❌ 2.00× slower           |
| Hamming distance, 200 pairs × 128 characters        |        80.7 µs |          15.9 µs | ❌ 5.08× slower           |
| Jaro similarity, 200 pairs × 128 characters         |         609 µs |           206 µs | ❌ 2.96× slower           |
| Jaro-Winkler similarity, 200 pairs × 128 characters |         613 µs |           206 µs | ❌ 2.98× slower           |
| Prefix distance, 200 pairs × 128 characters         |        13.0 µs |          12.5 µs | ❌ 1.04× slower           |
| Postfix distance, 200 pairs × 128 characters        |        12.4 µs |          12.4 µs | ≈ Same speed              |

### Fuzzy scorers

| Workload                                  | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ----------------------------------------- | -------------: | ---------------: | ------------------------- |
| `similarity`, 200 sentence pairs          |        42.0 µs |          47.9 µs | ✅ **1.14× faster**       |
| `partialSimilarity`, 200 sentence pairs   |         884 µs |           413 µs | ❌ 2.14× slower           |
| `tokenSortSimilarity`, 200 sentence pairs |         341 µs |           230 µs | ❌ 1.48× slower           |
| `tokenSetSimilarity`, 200 sentence pairs  |         548 µs |           286 µs | ❌ 1.91× slower           |
| `fuzzySimilarity`, 200 sentence pairs     |         790 µs |           404 µs | ❌ 1.96× slower           |

RapidFuzz's `QRatio` has no spelling in this API and is not measured. Its work
is `normalizeText` followed by `similarity`, both of which are.

### Search and batch scoring

| Workload                                                   | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ---------------------------------------------------------- | -------------: | ---------------: | ------------------------- |
| `bestMatch` + `similarity`, 20 queries × 2,000 choices     |        2.99 ms |          1.54 ms | ❌ 1.95× slower           |
| `bestMatch` + `tokenSortSimilarity`, 20 × 2,000 raw titles |        31.6 ms |          19.3 ms | ❌ 1.64× slower           |
| `Matcher` + `tokenSortSimilarity`, 20 × 2,000 titles       |        5.67 ms |          19.3 ms | ✅ **3.40× faster**       |
| `scoreMatrix` + `similarity`, 50 × 200                     |         642 µs |           235 µs | ❌ 2.74× slower           |
| `scoreMatrix` + `tokenSortSimilarity`, 50 × 200            |        1.77 ms |          4.30 ms | ✅ **2.43× faster**       |
| `scorePairs` + `similarity`, 200 pairs                     |        44.6 µs |          24.2 µs | ❌ 1.85× slower           |

### Edit operations

| Workload                                        | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ----------------------------------------------- | -------------: | ---------------: | ------------------------- |
| Levenshtein editops, 200 pairs × 128 characters |        1.32 ms |           326 µs | ❌ 4.07× slower           |
| LCS editops, 200 pairs × 128 characters         |         593 µs |           139 µs | ❌ 4.27× slower           |

### What the expanded comparison shows

`rapidfuzz-js` wins the smallest Levenshtein workload, `similarity`, the
token-sort matrix, and `Matcher` search. Postfix scanning is effectively tied at
this scale. Python RapidFuzz leads the longer Levenshtein cases, most other
distance metrics, the more complex fuzzy scorers, one-shot search, plain
similarity matrices, paired scoring, and edit operations.

At very small sizes, fixed call and loop overhead is a larger share of the
work, and V8 can optimize the surrounding JavaScript loop together with the
scorer. Python RapidFuzz pulls ahead as distance work grows because its kernels
and process operations execute inside a compiled C++ extension. In this corpus,
the Levenshtein crossover occurs between 8 and 32 characters; that boundary is
specific to these inputs and this machine.

The advantage is not uniform across batch operations. Python is 2.74× faster
for a plain `similarity` matrix, while `rapidfuzz-js` is 2.43× faster for the
multiword `tokenSortSimilarity` matrix. Different scorers move different amounts
of normalization, tokenization, and reusable setup into the batch path.

### A Matcher changes the result

The two token-search rows use the same 20 queries, 2,000 titles, scorer, and
Python measurement. Searching one-shot, Python is 1.64× faster. Reusing a
`Matcher` changes the outcome and makes `rapidfuzz-js` 3.40× faster than Python.
The paired same-process comparison above measures the improvement itself at
5.14×.

The `Matcher` is constructed before the timed search loop, representing a stable
catalog searched repeatedly. Its construction and retained memory are not
included in that row; the internal table above states construction separately.

Python RapidFuzz internally prepares a query within one `process` call, much as
`rapidfuzz-js` does for a one-shot search. It does not expose an equivalent
persistent prepared collection, so it reprocesses the title collection for each
query in this comparison.

### Comparison limits

- JavaScript and Python run in separate processes, although they use the same
  corpus and timing procedure.
- Corpus generation and JSON loading happen outside the timed workloads.
- Each workload receives three warm-up passes, is calibrated to at least 50 ms,
  and reports the median of nine passes.
- Results depend on string length, similarity, alphabet, cutoff, scorer, runtime
  versions, and hardware.
- Raw times are useful for this machine only; relative results are more
  portable, but should still be reproduced on the target system.

Python RapidFuzz provides higher throughput for most general distance and edit
workloads when its C++ extension is available. `rapidfuzz-js` is competitive on
small scorers and can lead token-heavy batch workflows when a collection is held
in a `Matcher`. It also runs directly in Node.js, browsers, and edge runtimes
without crossing into a Python service.

## Methodology

The cross-library runner is designed around relative measurements:

1. It creates deterministic, seeded input data.
2. It checks exact Levenshtein implementations for matching results.
3. It runs all JavaScript contenders in one process so they share the JIT,
   heap, and thermal state.
4. It gives each workload three warm-up passes.
5. It repeats work until a timed pass lasts at least 50 ms.
6. It records the median of nine timed passes.

The corpus contains:

- 200 edited string pairs at lengths 8, 32, and 128;
- 25 edited pairs at length 1,024;
- 200 sentence pairs;
- 2,000 single-word choices and 20 queries;
- 2,000 five-word titles and 20 multiword queries; and
- 50 × 200 score matrices.

Inputs are similar strings with seeded edits rather than unrelated random
strings, so the distance kernels do meaningful work instead of taking only
early exits.

Source files:

- [`bench/comparison/run.mjs`](bench/comparison/run.mjs) runs the comparisons.
- [`bench/comparison/corpus.mjs`](bench/comparison/corpus.mjs) builds the corpus.
- [`bench/comparison/timing.mjs`](bench/comparison/timing.mjs) defines the
  JavaScript timing loop.
- [`bench/comparison/rapidfuzz_bench.py`](bench/comparison/rapidfuzz_bench.py)
  mirrors that loop for Python.

## Internal performance metrics

The internal suite answers a different question: whether a code change makes
this library faster or slower than its stored baseline. It covers distance
metrics, edit operations, fuzzy scorers, one-shot search, `Matcher`
construction and reuse, matrices, paired scoring, cutoffs, and Unicode cases.

`bench/tooling/compare.ts` normalizes every result against control workloads
measured before and after the suite. This reduces distortion from CPU
frequency, background load, and thermal changes.

Important fields in
[`bench/tooling/baseline.json`](bench/tooling/baseline.json):

| Field                | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| `median`             | Median case time in milliseconds                          |
| `machine`            | Control timings for the same session, by control name     |
| `normalised`         | Case time divided by that session's machine speed         |
| `noise`              | Relative spread across repeated normalized measurements   |
| `samples`            | Number of samples collected                               |
| `source`             | Fingerprint of the benchmark and shared measurement files |
| `environment` fields | Node, platform, CPU, and measurement versions             |

A case is reported as changed only when it moves beyond its combined current
and baseline noise band, with a minimum threshold of 3%. Quick comparisons use
a wider 15% threshold. The runner also rejects excessive machine drift and
detects broad suite-wide movement.

## Run the benchmarks

Install the root dependencies first:

```sh
pnpm install
```

### Quick feedback while editing

```sh
pnpm bench:quick
pnpm bench:quick bench/fuzz.bench.ts
pnpm bench:quick -t 'partialRatio'
```

Quick mode uses shorter warm-up and measurement windows. It can detect large
changes, but its numbers should not support precise performance claims.

### Compare with the stored baseline

```sh
pnpm bench:compare
pnpm bench:compare:quick
pnpm bench:compare --fail-on-regression
pnpm bench:compare:quick -t 'indelDistance'
```

Use the full comparison before accepting or publishing a performance change.
To intentionally replace the stored baseline after review:

```sh
pnpm bench:baseline
```

### Compare with other libraries

```sh
pnpm build
pnpm install --dir bench/comparison
pnpm bench:libraries
```

The runner writes raw results to `bench/comparison/last-run.json`, which is
ignored by Git.

To include Python RapidFuzz:

```sh
python3 -m venv .venv
.venv/bin/pip install rapidfuzz numpy
node bench/comparison/run.mjs --python=.venv/bin/python
```

You can also set `RAPIDFUZZ_PYTHON` to the Python interpreter path.

## Interpreting your own run

- Compare ratios, not raw times, across machines.
- Treat differences inside the reported noise band as unchanged.
- Rerun results marked with high spread or machine drift.
- Compare scoring libraries semantically before comparing their speed.
- Benchmark representative string lengths, similarity levels, cutoffs, and
  query-to-choice ratios for your application.
