# Benchmarks

This document explains how `rapidfuzz-js` performs, how the measurements are
calculated, and how to reproduce them.

## Summary

The recorded comparison shows that `rapidfuzz-js`:

- is close to the fastest specialized JavaScript Levenshtein libraries on
  short strings and faster on longer strings;
- is 3–20× faster than `fuzzball` for the measured Levenshtein workloads;
- is about 16× faster than `fuzzball` for `ratio` and 5× faster for best-match
  search;
- makes repeated scoring 1.19–7.22× faster in the measured prepared-query and
  prepared-choice workloads;
- makes prepared token-sort search 2.74× faster than Python RapidFuzz and
  16.92× faster than `fuzzball` on the measured stable-catalog workload; and
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
| 8 characters, 200 pairs    | ❌ 1.06× slower       | ✅ **1.42× faster**  | ❌ 1.70× slower     | ✅ **2.97× faster**  |
| 32 characters, 200 pairs   | ✅ **1.09× faster**   | ✅ **3.10× faster**  | ✅ **1.94× faster** | ✅ **11.69× faster** |
| 128 characters, 200 pairs  | ✅ **1.36× faster**   | ✅ **9.56× faster**  | ✅ **5.22× faster** | ✅ **15.53× faster** |
| 1,024 characters, 25 pairs | ✅ **1.62× faster**   | ✅ **18.66× faster** | ✅ **9.84× faster** | ✅ **19.83× faster** |

For eight-character inputs, `fastest-levenshtein` and `js-levenshtein` were
faster. `rapidfuzz-js` moved ahead of `fastest-levenshtein` at 32 characters
and widened the lead as input length increased.

### Similarity and best-match search

| Workload                             | Compared with                   | Result for `rapidfuzz-js` |
| ------------------------------------ | ------------------------------- | ------------------------- |
| `ratio`, 200 sentence pairs          | `fuzzball`                      | ✅ **16.35× faster**      |
| `ratio`, 200 sentence pairs          | `string-similarity`             | ✅ **27.43× faster**      |
| Best of 2,000 choices for 20 queries | `fuzzball`                      | ✅ **5.34× faster**       |
| Best of 2,000 choices for 20 queries | `string-similarity`             | ✅ **21.87× faster**      |
| Best of 2,000 choices for 20 queries | `fuse.js` with a prebuilt index | ✅ **62.61× faster**      |

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

## Prepared inputs: why they matter

Most scorers do more than calculate the final score. Depending on the scorer
and processor, each call may normalize text, split and sort tokens, deduplicate
tokens, build character masks, or create another query representation. When an
operand is reused, repeating that setup produces the same intermediate data
again.

`rapidfuzz-js` can retain that reusable work:

- `prepareQuery` prepares one query for many choices.
- `prepareChoice` prepares one choice for many queries.
- `prepareChoices` prepares a complete collection for repeated `extract*`
  searches.
- A prepared query and prepared choice can be composed so neither operand is
  prepared again.

### How much work is avoided

For `Q` queries and `C` choices, the conceptual setup counts are:

| Approach                         | Query setup | Choice setup | Pair scores |
| -------------------------------- | ----------: | -----------: | ----------: |
| Direct pair-scoring loop         |     `Q × C` |      `Q × C` |     `Q × C` |
| `prepareQuery`                   |         `Q` |      `Q × C` |     `Q × C` |
| `prepareChoice` for every choice |     `Q × C` |          `C` |     `Q × C` |
| Both halves prepared             |         `Q` |          `C` |     `Q × C` |
| `extract*` with raw choices      |         `Q` |      `Q × C` |     `Q × C` |
| `extract*` with `prepareChoices` |         `Q` |          `C` |     `Q × C` |

Preparation does not skip required comparisons. It removes repeated setup
around those comparisons. This is why the benefit grows when an input is
reused many times and when its scorer performs expensive token or mask setup.

### Measured effect

The internal benchmark pairs every prepared path with the direct loop it
replaces:

| Prepared path                         | Workload                 | Result              |
| ------------------------------------- | ------------------------ | ------------------- |
| `prepareQuery` + `ratio`              | 1 query × 200 choices    | ✅ **2.08× faster** |
| `prepareQuery` + `tokenSortRatio`     | 1 query × 200 sentences  | ✅ **1.99× faster** |
| `prepareQuery` + `wRatio`             | 1 query × 200 sentences  | ✅ **1.64× faster** |
| `prepareChoice` + `ratio` + processor | 20 queries × 200 choices | ✅ **1.81× faster** |
| `prepareChoice` + `tokenSortRatio`    | 20 queries × 200 choices | ✅ **2.21× faster** |
| Both halves + `tokenSortRatio`        | 20 queries × 200 choices | ✅ **7.22× faster** |
| Both halves + `levenshteinDistance`   | 20 queries × 200 choices | ✅ **1.19× faster** |

These speedups use normalized values from the stored M1 Max baseline. Handle
construction is included wherever a caller would normally pay for it: once per
query, once per choice collection, or both.

The 7.22× result is the clearest example of preparation's value. In the direct
loop, both sides are tokenized repeatedly across 4,000 comparisons. With both
halves prepared, 20 queries and 200 choices are tokenized once each, while the
same 4,000 scores are still calculated. `levenshteinDistance` improves by only
1.19× because its reusable setup is a smaller part of the total work.

See [`bench/handles.bench.ts`](bench/handles.bench.ts) for the paired cases and
the [README examples](README.md#reuse-prepared-inputs) for usage.

### How this differs from the comparison libraries

The pinned library versions in this benchmark expose these reuse options:

| Library               | Public reuse mechanism                  | Practical difference                                                                           |
| --------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `rapidfuzz-js`        | Prepared query, choice, or collection   | Reuses either or both scorer operands without changing the scorer                              |
| `fastest-levenshtein` | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                        |
| `leven`               | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                        |
| `js-levenshtein`      | No prepared-operand API                 | Exposes pair scoring only                                                                      |
| `fuzzball`            | No prepared-operand API                 | Scorers and `extract` process inputs within each operation                                     |
| `string-similarity`   | No prepared-operand API                 | Rebuilds bigrams during `compareTwoStrings` calls                                              |
| `fuse.js`             | Prebuilt collection index               | Reuses collection state, but uses Bitap and does not expose prepared scorer operands           |
| Python RapidFuzz      | Internal query preparation in `process` | Reuses a query within one search call, but does not expose persistent query and choice handles |

This does not mean every comparison table should be multiplied by the prepared
speedups. The cross-library Levenshtein cases score independent pairs, where
there is no reusable operand. The similarity and search comparison uses raw
`ratio` inputs with preprocessing disabled, while the largest prepared gains
come from repeated token scoring or processing. Those are different workloads.

Fuse is also not an unindexed comparison: its index is built before the timed
search loop. Its 62.61× loss in the search table already compares a prebuilt Fuse
index with an unindexed `rapidfuzz-js` scan, although the two libraries use
different matching algorithms.

Prepared inputs are most useful for stable data searched or joined repeatedly,
such as autocomplete catalogs, deduplication batches, record linkage, and
many-query ranking. For a one-off pair, preparation adds setup that may not be
recovered. Prepared operands must also be rebuilt after their source data
changes.

### Preparation versus other libraries

Two cross-library workloads isolate how preparation changes the result.

For a fixed 128-character Levenshtein query scored against 200 choices:

| Comparison            | External time | Prepared `rapidfuzz-js` | Preparation gain | Overall result       |
| --------------------- | ------------: | ----------------------: | ---------------: | -------------------- |
| `rapidfuzz-js` direct |      0.411 ms |                0.341 ms |     ✅ **1.20×** | Preparation alone    |
| `fastest-levenshtein` |      0.905 ms |                0.321 ms |                — | ✅ **2.82× faster**  |
| `leven`               |       9.19 ms |                0.319 ms |                — | ✅ **28.78× faster** |
| `js-levenshtein`      |       4.45 ms |                0.319 ms |                — | ✅ **13.95× faster** |
| `fuzzball`            |       9.13 ms |                0.317 ms |                — | ✅ **28.84× faster** |
| Python RapidFuzz      |      0.280 ms |                0.324 ms |                — | ❌ 1.15× slower      |

Preparation makes the JavaScript path 1.20× faster than its own direct call. It
also narrows the Python gap from 1.47× slower to 1.15× slower. The much larger
leads over the JavaScript competitors combine this preparation benefit with
`rapidfuzz-js`'s bit-parallel Levenshtein implementation; preparation alone
does not explain a 28× difference.

Token-sort search shows a larger preparation effect because tokenization and
sorting can be reused for the whole collection. The paired JavaScript and
`fuzzball` pass measures preparation directly:

| 20 queries × 2,000 titles       |    Time | Result                              |
| ------------------------------- | ------: | ----------------------------------- |
| `rapidfuzz-js`, raw titles      | 30.9 ms | Baseline                            |
| `rapidfuzz-js`, prepared titles | 6.24 ms | ✅ **4.95× faster than raw**        |
| `fuzzball`, raw titles          |  107 ms | ✅ **prepared JS is 16.92× faster** |

The separately sequenced Python pass compares both JavaScript modes with the
same Python measurement:

| 20 queries × 2,000 titles       |    Time | Result versus Python |
| ------------------------------- | ------: | -------------------- |
| `rapidfuzz-js`, raw titles      | 67.3 ms | ❌ 3.52× slower      |
| `rapidfuzz-js`, prepared titles | 6.98 ms | ✅ **2.74× faster**  |
| Python RapidFuzz, raw titles    | 19.1 ms | Reference            |

The raw JavaScript time moved between the same-process and cross-runtime passes,
which is why the preparation gain is taken from the paired 30.9 ms versus
6.24 ms measurement. This avoids presenting timing drift as preparation
speedup.

Neither `fuzzball` nor Python RapidFuzz exposes a persistent prepared-choice
collection for this scorer. They prepare a query within a search operation but
process the title collection again for each new query. `prepareChoices` pays
that title-side work once and reuses it across the 20 searches.

`string-similarity` and Fuse are not included in this prepared token table
because they do not implement the same token-sort scorer. Fuse's prebuilt Bitap
index is covered separately in the best-match table; comparing it here would
mix preparation gains with a different matching algorithm.

## Python RapidFuzz comparison

The Python leg uses the same seeded corpus, loop shapes, warm-up count, pass
length, and median statistic as the JavaScript leg. Times are medians for the
complete workload in each row, not for one string pair.

### Distance and similarity metrics

| Workload                                            | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| --------------------------------------------------- | -------------: | ---------------: | ------------------------- |
| Levenshtein, 200 pairs × 8 characters               |        13.4 µs |          24.3 µs | ✅ **1.82× faster**       |
| Levenshtein, 200 pairs × 32 characters              |        39.6 µs |          34.0 µs | ❌ 1.17× slower           |
| Levenshtein, 200 pairs × 128 characters             |         395 µs |           227 µs | ❌ 1.74× slower           |
| Levenshtein, 25 pairs × 1,024 characters            |        2.24 ms |          1.03 ms | ❌ 2.18× slower           |
| Fixed-query Levenshtein, direct, 1 × 200 × 128      |         413 µs |           280 µs | ❌ 1.47× slower           |
| Fixed-query Levenshtein, prepared, 1 × 200 × 128    |         324 µs |           280 µs | ❌ 1.15× slower           |
| Indel distance, 200 pairs × 128 characters          |         335 µs |          69.1 µs | ❌ 4.85× slower           |
| LCSseq similarity, 200 pairs × 128 characters       |         334 µs |          68.8 µs | ❌ 4.86× slower           |
| OSA distance, 200 pairs × 128 characters            |         759 µs |           251 µs | ❌ 3.02× slower           |
| Damerau-Levenshtein, 200 pairs × 128 characters     |        10.9 ms |          5.42 ms | ❌ 2.01× slower           |
| Hamming distance, 200 pairs × 128 characters        |        80.2 µs |          15.5 µs | ❌ 5.15× slower           |
| Jaro similarity, 200 pairs × 128 characters         |         613 µs |           201 µs | ❌ 3.05× slower           |
| Jaro-Winkler similarity, 200 pairs × 128 characters |         625 µs |           202 µs | ❌ 3.09× slower           |
| Prefix distance, 200 pairs × 128 characters         |        12.8 µs |          12.3 µs | ❌ 1.04× slower           |
| Postfix distance, 200 pairs × 128 characters        |        12.2 µs |          12.2 µs | ≈ Same speed              |

### Fuzzy scorers

| Workload                             | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ------------------------------------ | -------------: | ---------------: | ------------------------- |
| `ratio`, 200 sentence pairs          |        41.5 µs |          47.3 µs | ✅ **1.14× faster**       |
| `partialRatio`, 200 sentence pairs   |        1.02 ms |           412 µs | ❌ 2.46× slower           |
| `tokenSortRatio`, 200 sentence pairs |         355 µs |           225 µs | ❌ 1.58× slower           |
| `tokenSetRatio`, 200 sentence pairs  |         559 µs |           270 µs | ❌ 2.07× slower           |
| `wRatio`, 200 sentence pairs         |         794 µs |           395 µs | ❌ 2.01× slower           |
| `qRatio`, 200 sentence pairs         |        44.2 µs |          47.3 µs | ✅ **1.07× faster**       |

### Search and batch scoring

| Workload                                               | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ------------------------------------------------------ | -------------: | ---------------: | ------------------------- |
| `extractOne` + `ratio`, 20 queries × 2,000 choices     |        2.77 ms |          1.53 ms | ❌ 1.81× slower           |
| `extractOne` + `tokenSortRatio`, 20 × 2,000 raw titles |        67.3 ms |          19.1 ms | ❌ 3.52× slower           |
| `extractOne` + `tokenSortRatio`, prepared JS titles    |        6.98 ms |          19.1 ms | ✅ **2.74× faster**       |
| `scoreMatrix` + `ratio`, 50 × 200                      |         782 µs |           231 µs | ❌ 3.39× slower           |
| `scoreMatrix` + `tokenSortRatio`, 50 × 200             |        2.13 ms |          4.23 ms | ✅ **1.99× faster**       |
| `scorePairs` + `ratio`, 200 pairs                      |        45.9 µs |          23.6 µs | ❌ 1.95× slower           |

### Edit operations

| Workload                                        | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ----------------------------------------------- | -------------: | ---------------: | ------------------------- |
| Levenshtein editops, 200 pairs × 128 characters |        1.32 ms |           326 µs | ❌ 4.06× slower           |
| LCSseq editops, 200 pairs × 128 characters      |         598 µs |           137 µs | ❌ 4.35× slower           |

### What the expanded comparison shows

`rapidfuzz-js` wins the smallest Levenshtein workload, `ratio`, `qRatio`, the
token-sort matrix, and prepared token search. Prefix and postfix scanning are
effectively tied at this scale. Python RapidFuzz leads the longer Levenshtein
cases, most other distance metrics, the more complex fuzzy scorers, raw search,
plain ratio matrices, paired scoring, and edit operations.

At very small sizes, fixed call and loop overhead is a larger share of the
work, and V8 can optimize the surrounding JavaScript loop together with the
scorer. Python RapidFuzz pulls ahead as distance work grows because its kernels
and process operations execute inside a compiled C++ extension. In this corpus,
the Levenshtein crossover occurs between 8 and 32 characters; that boundary is
specific to these inputs and this machine.

The advantage is not uniform across batch operations. Python is 3.25× faster
for a plain `ratio` matrix, while `rapidfuzz-js` is 1.95× faster for the
multiword `tokenSortRatio` matrix. Different scorers move different amounts of
normalization, tokenization, and reusable setup into the batch path.

### Prepared search changes the result

The two token-search rows use the same 20 queries, 2,000 titles, scorer, and
Python measurement. With raw JavaScript titles, Python is 3.52× faster. Reusing
a `prepareChoices` index changes the outcome and makes `rapidfuzz-js` 2.74×
faster than Python. The paired same-process comparison above measures the
preparation improvement itself at 4.95×.

The prepared index is built before the timed search loop, representing a stable
catalog searched repeatedly. Its construction and retained memory are not
included in that row. The earlier internal prepared-input table includes handle
construction in workloads where a caller would pay it per batch.

Python RapidFuzz internally prepares a query within one `process` call, much as
`rapidfuzz-js` does for raw `extract*` calls. It does not expose an equivalent
persistent prepared-choice collection, so it reprocesses the title collection
for each query in this comparison.

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
small scorers and can lead token-heavy batch workflows when reusable state is
prepared. It also runs directly in Node.js, browsers, and edge runtimes without
crossing into a Python service.

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
metrics, edit operations, fuzzy scorers, search, matrices, cutoffs, prepared
inputs, and Unicode cases.

`bench/compare.mjs` normalizes every result against control workloads measured
before and after the suite. This reduces distortion from CPU frequency,
background load, and thermal changes.

Important fields in [`bench/baseline.json`](bench/baseline.json):

| Field                | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| `median`             | Median case time in milliseconds                          |
| `anchor`             | Control timing for the same run                           |
| `normalised`         | Case time divided by its control anchor                   |
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
