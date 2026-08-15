# Benchmarks

This document explains how `rapidfuzz-js` performs, how the measurements are
calculated, and how to reproduce them.

Every number below comes from a single recorded pass of
[`bench/comparison/libraries/run.mjs`](bench/comparison/libraries/run.mjs), plus one pass of the
internal `bench/suites/process.bench.ts` for the `Matcher` table. Rerunning replaces
all of them together; do not update one table from a later run.

Two sections are exceptions, and both are marked as such:

- The **n-gram** tables were added after that pass, so they come from a later
  one and carry no absolute times. Every figure in them is a ratio between two
  libraries measured in the same process on the same corpus, which is what
  makes them comparable to the rest without being from the same pass — and they
  were reproduced across three runs.
- **Searching a growing collection** is a separate script with its own corpus,
  `bench/comparison/indexedSearch/throughput.mjs`, because it varies the size of the
  collection and the rest of the suite does not. Its times are absolute, every
  arm in it ran in one process, and a second run reproduced every figure over
  1 ms to within 4% and the sub-microsecond cells to within 10%. Its memory
  table comes from `indexedSearch/memory.mjs`, which measures every cell in a
  child process of its own and reproduced identically across three runs, and
  its hit-rate table from `indexedSearch/quality.mjs`, which is deterministic — the
  corpus, the damage and every library's configuration are fixed, so it returns
  the same counts every time.

The next full re-record should fold the first of these in and drop that bullet.

## Summary

The recorded comparison shows that `rapidfuzz-js`:

- is close to the fastest specialized JavaScript Levenshtein libraries on
  short strings and faster on longer strings;
- is 2.9–19.5× faster than `fuzzball` for the measured Levenshtein workloads;
- is about 16× faster than `fuzzball` for `ratio` and 5× faster for
  best-match search;
- computes Dice 1.22–1.45× faster than the other multiset implementations and
  Cosine 2.57–3.39× faster than the only comparable one, losing only to
  `dice-coefficient` below about 64 characters;
- answers repeated queries over a stable collection 5.14× faster through a
  `Matcher` than through the same one-shot search, and scores a held query
  1.29× faster than scoring each pair directly;
- searches 100,000 choices for a misspelled entry 11.3× faster through
  `createIndexedMatcher` than through a prepared `Matcher`, and 78–519× faster
  than every other JavaScript package measured — including 1.28× faster than
  uFuzzy once uFuzzy is configured to find a typo at all;
- returns the same results as the exhaustive `Matcher` in every measured case,
  and comes within one case of the best library measured on every kind of typo,
  losing only on prefix queries, where a position-aware matcher is the right
  tool;
- retains 5.5× less memory in that index than in the prepared collection it
  replaces, and 7× less than the gram arrays `dice-coefficient` needs: 149 MB
  against 813 MB and 1,048 MB at 1,000,000 choices;
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
| `ratio`, 200 sentence pairs          | `fuzzball`                      | ✅ **16.11× faster**      |
| `ratio`, 200 sentence pairs          | `string-similarity`             | ✅ **27.99× faster**      |
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

### Sørensen-Dice and Cosine over n-grams

Recorded in a later pass than the tables above, so these are ratios only —
see the note at the top. Before timing, the runner checks that every contender
in the first table returns our number for every pair in the corpus. All four
compute Dice over a _multiset_ of bigrams, which is what `similarity` from
`rapidfuzz-js/dice` does; the n-gram checks use a tolerance rather than `!==`,
because Dice is a ratio of integers and comes back identical while cosine
divides by a square root.

| Input                      | `dice-coefficient` | `fast-dice-coefficient` | `string-similarity` | `string-comparison` |
| -------------------------- | ------------------ | ----------------------- | ------------------- | ------------------- |
| 8 characters, 200 pairs    | ❌ 2.51× slower    | ✅ **1.24× faster**     | ✅ **1.26× faster** | ✅ **1.29× faster** |
| 32 characters, 200 pairs   | ❌ 1.74× slower    | ✅ **1.22× faster**     | ✅ **1.28× faster** | ✅ **1.40× faster** |
| 128 characters, 200 pairs  | ✅ **1.44×**       | ✅ **1.30× faster**     | ✅ **1.36× faster** | ✅ **1.35× faster** |
| 1,024 characters, 25 pairs | ✅ **10.00×**      | ✅ **1.42× faster**     | ✅ **1.45× faster** | ✅ **1.39× faster** |

`dice-coefficient` is the only JavaScript library measured here that beats
`rapidfuzz-js` at its own metric, and the reason is structural rather than
incidental. It compares two bigram arrays with a nested scan, allocating
nothing: `O(n·m)` work with no setup, against our `O(n)` work behind a trie
that has to be built first. The build dominates while the strings are short,
the quadratic term dominates once they are not, and the crossover sits near 64
characters. For short words scored one pair at a time, it is the faster
choice.

Cosine has one true competitor: `wink-nlp-utils` bags the n-grams with their
counts and `wink-distance` takes the cosine of two such bags. The pair agrees
with `cosineSimilarity` to within 2e-16, and is the only other frequency-vector
n-gram cosine in JavaScript.

| Input                      | `wink` bag-of-n-grams + `bow.cosine` |
| -------------------------- | ------------------------------------ |
| 8 characters, 200 pairs    | ✅ **3.16× faster**                  |
| 32 characters, 200 pairs   | ✅ **3.19× faster**                  |
| 128 characters, 200 pairs  | ✅ **3.39× faster**                  |
| 1,024 characters, 25 pairs | ✅ **2.57× faster**                  |

At `gramSize: 3` over 128 characters, Dice measured 1.20× faster than
`dice-coefficient` and Cosine 3.35× faster than the wink pair.

For 20 queries against 2,000 choices, where a `Matcher` profiles each choice
once instead of once per query:

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

The `dice-coefficient` rows are the same story as the pairwise table at a
different scale: over 12-character choices its allocation-free scan keeps it
ahead of a one-shot `bestMatch`, and a `Matcher` passes it anyway by not
rebuilding the profiles.

Three contenders are measured by the suite but excluded from the tables above,
because they answer a different question:

| Library                    | What it computes                                     | Measured                 |
| -------------------------- | ---------------------------------------------------- | ------------------------ |
| `talisman/metrics/dice`    | Dice over a **set** of bigrams                       | 2.32–2.79× faster for us |
| `natural.DiceCoefficient`  | Dice over a **set** of bigrams                       | 1.45–1.60× faster for us |
| `string-comparison.cosine` | Binary vector over **characters**, not n-gram counts | 1.40–2.82× slower for us |

Set-based Dice scores `'aaaa'` against `'aaa'` as `1` where the multiset form
answers `0.8`, so those two do strictly less work. `string-comparison`'s
cosine is doubly different: it is binary rather than frequency-weighted, and
over characters rather than grams, which makes `'iwmaxzsz'` and `'iwmaxssz'`
score a flat `1`.

## Held inputs: what a Matcher reuses

Most scorers do more than calculate the final score. Depending on the scorer
and normalizer, each call may normalize text, split and sort tokens, deduplicate
tokens, build character masks, or create another query representation. When an
operand is reused, repeating that setup produces the same intermediate data
again.

`rapidfuzz-js` holds that reusable work in four places, and none of them is a
handle the caller has to manage:

- A one-shot `bestMatch`, `search` or `searchIter` call prepares the query once
  and reuses it across every choice in that call.
- `scoreMatrix` prepares every row once and every column once, so an `R × C`
  matrix costs `R + C` preparations rather than `R × C`.
- A `Matcher` prepares every choice when it is constructed and reuses those
  preparations for every later `best`, `search` and `searchIter` call.
- An indexed `Matcher` — `createIndexedMatcher`, for the n-gram scorers — goes
  further and holds the collection as one inverted structure, so a query visits
  only the choices sharing an n-gram with it instead of all of them.
  [Searching a growing collection](#searching-a-growing-collection) measures it.

A collection is held by constructing a `Matcher`, and a query is held for the
length of the call that uses it, so neither needs a handle the caller manages.
`scorer.prepareChoice` is the one public handle: it holds a single choice for
callers who keep the collection themselves, and the figures below do not
measure it.

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

| Workload                                    | One-shot | `Matcher` | Result              |
| ------------------------------------------- | -------: | --------: | ------------------- |
| 30 queries × 2,000 choices, `ratio`         |  5.09 ms |   3.53 ms | ✅ **1.44× faster** |
| 30 queries × 2,000 titles, `tokenSortRatio` |  64.4 ms |   9.71 ms | ✅ **6.63× faster** |

Construction is what buys that, and it is paid once: 0.058 ms for the 2,000
single-word choices, and 0.939 ms for the 2,000 five-word titles with
normalization. The token-sort case gains most because tokenizing and sorting
2,000 titles is repeated for every query in the one-shot loop and done once for
the `Matcher`. Plain `ratio` gains least because its reusable setup is a
smaller share of the total work.

See [`bench/suites/process.bench.ts`](bench/suites/process.bench.ts) for the paired cases and
the [README](README.md#one-query-or-many) for usage. The suite does not yet
carry a one-shot partner for the normalized-edit `Matcher` case, so that family
has no paired figure here.

### How this differs from the comparison libraries

The pinned library versions in this benchmark expose these reuse options:

| Library               | Public reuse mechanism                  | Practical difference                                                                        |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------- |
| `rapidfuzz-js`        | `Matcher` over a collection             | Prepares every choice once at construction and reuses it for every later query              |
| `rapidfuzz-js`        | Indexed `Matcher`, n-gram scorers       | Holds the collection as one inverted structure, so a query skips choices it cannot match    |
| `fastest-levenshtein` | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                     |
| `leven`               | No prepared-operand API                 | Pair and closest-match calls build their working state during each call                     |
| `js-levenshtein`      | No prepared-operand API                 | Exposes pair scoring only                                                                   |
| `fuzzball`            | No prepared-operand API                 | Scorers and `extract` process inputs within each operation                                  |
| `string-similarity`   | No prepared-operand API                 | Rebuilds bigrams during `compareTwoStrings` calls                                           |
| `fuse.js`             | Prebuilt collection index               | Reuses collection state, but uses Bitap and does not expose prepared scorer operands        |
| `uFuzzy`              | None, by design                         | Builds nothing and scans the haystack with a compiled RegExp on every query                 |
| Python RapidFuzz      | Internal query preparation in `process` | Reuses a query within one search call, but does not expose a persistent prepared collection |

This does not mean every comparison table should be multiplied by the `Matcher`
speedups. The cross-library Levenshtein cases score independent pairs, where
there is no reusable operand. The similarity and search comparison uses raw
`ratio` inputs with preprocessing disabled, while the largest `Matcher`
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

## Searching a growing collection

Every table above searches 2,000 choices. This one varies the size of the
collection — 100 to 1,000,000 — and asks the question a search box asks: _which
of these strings best matches the one I typed_. It is where the two matchers
separate: a `Matcher` scores every choice, and an indexed one scores only the
choices that share an n-gram with the query.

Recorded in its own pass of
[`bench/comparison/indexedSearch/throughput.mjs`](bench/comparison/indexedSearch/throughput.mjs) and
reproduced in a second run; see the note at the top of this document.

### The workload

Choices are four-word phrases from a Zipf-weighted vocabulary, so a few words
carry most of the text as they do in real prose — a uniform corpus would give
every n-gram the same posting length and flatter an index. The generator is
seeded and drawn in one pass, so a smaller corpus is a prefix of a larger one:
the ladder varies `N` and nothing else. Every arm is asked for the best 5 above
a Dice similarity of `0.5`, at the default `gramSize: 2`.

Three arms are ours — a one-shot `search` holding nothing, a `createMatcher`
holding every choice's grams, and a `createIndexedMatcher` holding one inverted
structure over the whole collection. Five are other packages:

- `dice-coefficient` handed prebuilt gram arrays: the only other package here
  that holds anything, and the fastest scan in the field;
- `string-similarity.findBestMatch` and `fuzzball.extract`, the two that ship a
  search API, both of which reprocess the collection on every query;
- uFuzzy and Fuse, which build a different kind of answer entirely.

The three of ours and `dice-coefficient` compute one number, checked against
each other to 1e-12 before anything is timed. `string-similarity` is Dice over
bigrams too, but strips whitespace first, so it scores multi-word text slightly
differently — near enough to time against, not near enough to check.

**uFuzzy and Fuse are not a like-for-like.** uFuzzy matches a subsequence — the
query's characters in order, with bounded gaps — and Fuse is Bitap; both rank by
where a match landed, where Dice measures n-gram overlap and ignores position,
so `'new york mets'` and `'mets new york'` are near-identical to us and
unrelated to them. What compares is the time to narrow `N` candidates to a
handful, not the answer.

### Time for one query

Milliseconds for the `one typo` query — a catalog entry with one character
changed, which is what a fuzzy search is for. Median of nine passes; lower is
better.

| Arm                                |        100 |     1,000 |    10,000 |  100,000 | 1,000,000 |
| ---------------------------------- | ---------: | --------: | --------: | -------: | --------: |
| `createIndexedMatcher`             | **0.0021** | **0.013** | **0.219** | **1.76** |  **20.6** |
| `createMatcher`                    |     0.0097 |     0.207 |      2.10 |     19.8 |         — |
| `search`, one-shot                 |      0.147 |      1.72 |      15.3 |      182 |         — |
| `dice-coefficient`, prebuilt grams |      0.364 |      3.96 |      38.4 |      367 |         — |
| `string-similarity.findBestMatch`  |      0.318 |      3.64 |      36.0 |      331 |         — |
| `fuzzball.extract`                 |      0.129 |      1.56 |      14.9 |      137 |         — |
| `uFuzzy.search`, single error      |      0.026 |     0.084 |     0.243 |     2.26 |      86.4 |
| `Fuse`, prebuilt index             |      0.919 |      11.1 |      98.0 |      913 |         — |
| _`uFuzzy.filter`, defaults_        |   _0.0027_ |   _0.017_ |   _0.204_ |   _1.84_ |    _26.4_ |

Every arm but the index and uFuzzy was capped at 100,000 choices, which is what
the empty cells are: a 1,000,000-choice scan at 350 ms a query measures patience.

The last row is in italics because it is not doing the same job, and the gap
between it and the row above is the reason this table has two uFuzzy entries.
uFuzzy's defaults tolerate **no error inside a word** (`intraMode: 0`), and
`filter` narrows without ranking, so that row is the cost of returning an
unranked list that contains none of the typo's matches. `uFuzzy.search` with its
own single-error preset is the comparable arm: same job, ranked, typo found.
[Do they find it?](#do-they-find-it) measures that difference rather than
asserting it.

Read down a column. The order barely changes with size: the index and uFuzzy
first, the prepared collection next, then the two scans that reuse nothing much
— our one-shot `search` and `fuzzball.extract` trade places as `N` grows — and
the per-query rebuilders last.

Note also what the index does _not_ do here: its own time still grows with the
collection, because on this corpus the number of choices sharing grams with a
query grows with it too — 19 qualifying matches at 1,000 choices, 1,512 at
100,000. Indexing cuts the constant by an order of magnitude against a prepared
collection and by two against every other package; it does not make the size of
the collection stop mattering. Where it does change the shape is a query that
matches little, which is the last table in this section.

### At 100,000 choices

| Compared with                      | Result for `createIndexedMatcher` |
| ---------------------------------- | --------------------------------- |
| `createMatcher`, ours              | ✅ **11.3× faster**               |
| `search` one-shot, ours            | ✅ **103× faster**                |
| `fuzzball.extract`                 | ✅ **78× faster**                 |
| `string-similarity.findBestMatch`  | ✅ **188× faster**                |
| `dice-coefficient`, prebuilt grams | ✅ **209× faster**                |
| `Fuse`, prebuilt index             | ✅ **519× faster**                |
| `uFuzzy.search`, single error      | ✅ **1.28× faster**               |

The `createMatcher` row is the one to read first, because it is the same library
against itself: preparing the collection is worth 6.9× against the fastest
package that does not (`fuzzball`), and indexing it is worth another 11.3×.

### What the index costs, and when it pays

Setup is one measurement rather than a median — a collection is built once — so
read these as magnitudes, not as figures to compare at 10%:

| Held structure                |   100 | 1,000 | 10,000 | 100,000 | 1,000,000 |
| ----------------------------- | ----: | ----: | -----: | ------: | --------: |
| `createIndexedMatcher`        | 0.775 |  2.92 |   24.7 |     285 |     2,776 |
| `createMatcher`               | 0.290 |  1.47 |   16.7 |     182 |         — |
| `dice-coefficient` gram array | 0.095 | 0.225 |   1.86 |    46.0 |         — |
| Fuse index                    | 0.415 | 0.981 |   1.57 |    10.3 |         — |

An index costs more to build than a prepared collection — 1.6× at 100,000
choices, closer to 3× at 100 — and that is the whole of its downside. What it
buys back per query decides how many queries repay it:

- **100,000 choices.** It costs 103 ms more to build than a `Matcher` and saves
  18.0 ms a query, so **six queries** pay the difference. Against a one-shot
  `search`, which builds nothing at all, the entire 285 ms is back after **two**.
- **100 choices.** 0.485 ms more than a `Matcher`, saving 0.0076 ms a query:
  **64 queries**. Against a one-shot `search`, **five**.

So the index is not a large-collection feature so much as a repeated-query one.
The one shape it does not suit is a small list searched once or twice, where a
one-shot `search` builds nothing and is done before either constructor returns.

### What each structure retains

Holding a collection costs memory for as long as it is held, and that is the
half of the trade a timing table cannot show. Retained bytes per choice, with
the corpus itself charged to the baseline so each row is the structure alone:

| Held structure                 |  1,000 | 10,000 | 100,000 | 1,000,000 |
| ------------------------------ | -----: | -----: | ------: | --------: |
| `createIndexedMatcher`         |  216 B |  112 B |   147 B |     149 B |
| `createMatcher`                |  869 B |  822 B |   814 B |     813 B |
| `dice-coefficient` gram arrays | 1039 B | 1049 B |  1048 B |    1048 B |
| Fuse index                     |  260 B |   94 B |    74 B |      72 B |
| the choices themselves         |   96 B |   54 B |    54 B |      55 B |

At 1,000,000 choices that is **149 MB** for the index against **813 MB** for the
prepared collection and **1,048 MB** for the gram arrays `dice-coefficient`
needs, on top of 55 MB of strings. The index is the cheaper structure by 5.5× at
both 100,000 and 1,000,000 choices, so it is not only the faster of the two
matchers but the smaller one, which is unusual enough to be worth stating
plainly. Read the 1,000-choice column with suspicion: fixed overhead that does
not grow with the collection is a third of the index cell and nearly half the
corpus one, which is why both fall as `N` rises and then stay flat.

uFuzzy has no row because it retains nothing between queries — that is its
trade, and the per-query column above is where it pays for it. Fuse is the
opposite of a warning: it holds the least of any real structure here and is
still the slowest arm to search, because what it retains for a list of plain
strings is a light record per item rather than anything that narrows a query.

These figures are heap deltas, which have lied here before — measuring several
structures in one process has produced negative retained bytes, whatever the
previous arm was still collecting. Each cell is therefore measured in its own
child process, over a seeded corpus, and the whole table reproduced byte for
byte across three runs.

### The query decides more than the size does

The index visits only candidates sharing an n-gram with the query, so what it
saves depends on how many candidates that is — which the query decides as much
as `N` does. At 100,000 choices:

| Query                   |    Index | `Matcher` | Index gain |
| ----------------------- | -------: | --------: | ---------- |
| whole phrase            |  1.81 ms |   18.4 ms | 10.2×      |
| one typo                |  1.76 ms |   19.8 ms | 11.3×      |
| half a phrase           |  1.06 ms |   4.10 ms | 3.9×       |
| a common word           |  1.32 ms |   2.83 ms | 2.1×       |
| unrelated to the corpus | 0.162 ms |   15.3 ms | 94×        |

The two ends explain the mechanism. A query sharing nothing with the corpus
names no postings at all and is answered in microseconds, where a `Matcher` has
to look at all 100,000 choices to establish the same. A common short word is the
opposite: its few grams name much of the corpus, so the index touches nearly
everything — and it is also the case where the exhaustive path is at its
strongest, because a short query lets Dice's length bound reject most choices
before scoring them. 2.1× is the floor here, not a regression.

### Are they even the same algorithm?

No, and the differences decide when each one is the right tool. Speed is only
comparable between two libraries that would return an acceptable answer:

| Library                                     | What it computes                                                     | Position matters | Finds a typo           |
| ------------------------------------------- | -------------------------------------------------------------------- | ---------------- | ---------------------- |
| `rapidfuzz-js` `dice` / `cosine`            | overlap of two n-gram multisets                                      | no               | yes                    |
| `dice-coefficient`, `fast-dice-coefficient` | the same, over bigrams                                               | no               | yes                    |
| `string-similarity`                         | the same, with whitespace stripped first                             | no               | yes                    |
| `fuzzball.ratio`                            | Indel-normalized similarity — a RapidFuzz port, like ours            | yes              | yes                    |
| `uFuzzy`                                    | subsequence match with a per-term error budget, then its own ranking | yes              | only at `intraMode: 1` |
| `Fuse`                                      | Bitap approximate substring, scored by edit distance and location    | yes              | yes                    |

The three of ours and the three Dice packages are the same algorithm, so those
rows are a pure implementation comparison. `fuzzball` is a different metric from
the same family as ours. uFuzzy and Fuse are a different kind of tool: they ask
where a pattern occurs inside a string, not how much two strings overlap.

### Do they find it?

A search that answers in a microsecond without the entry the user meant has not
done the job, so the field is also measured on that. Forty entries are taken out
of the 10,000-choice corpus and damaged the way a person mistypes — one letter
substituted, one dropped, two swapped, the words reordered, or only half the
phrase remembered — and each library is asked for its best five. Every character
edit lands inside a word rather than on a space, which would otherwise measure a
corner of uFuzzy's term splitting instead of its matcher.

Each cell is **hit@1 / hit@5**: how often the entry the query came from was
ranked first, and how often it was in the five a UI would show, out of 40.

| Library                             | exact   | substitution | deletion | transposition | reordered | half a phrase |
| ----------------------------------- | ------- | ------------ | -------- | ------------- | --------- | ------------- |
| `createIndexedMatcher`              | 40/40   | 40/40        | 40/40    | 40/40         | 37/40     | 14/21         |
| `createMatcher`                     | 40/40   | 40/40        | 40/40    | 40/40         | 37/40     | 14/21         |
| `dice-coefficient` scan             | 40/40   | 40/40        | 40/40    | 40/40         | 37/40     | 14/21         |
| `string-similarity`                 | 40/40   | 40/40        | 40/40    | 40/40         | 38/40     | 15/22         |
| `fuzzball.extract`                  | 40/40   | 40/40        | 40/40    | 40/40         | 20/27     | 10/22         |
| `uFuzzy`, single error + reordering | 40/40   | 40/40        | 40/40    | 40/40         | 37/40     | 18/31         |
| `Fuse`, defaults                    | 40/40   | 40/40        | 40/40    | 40/40         | 23/32     | 21/32         |
| `Fuse`, `ignoreLocation`            | 40/40   | 40/40        | 40/40    | 40/40         | 23/29     | 16/28         |
| _`uFuzzy`, defaults_                | _40/40_ | _0/0_        | _0/0_    | _2/2_         | _0/0_     | _18/31_       |

Four things worth taking from it:

- **The two matchers are the same search.** Every cell of the indexed row equals
  the exhaustive one, which is the contract `createIndexedMatcher` is written to
  and what its tests pin. Indexing buys speed and costs nothing in results.
- **`dice-coefficient` returns our answers exactly**, as it should — same metric
  — and takes 175× longer to do it. Between two libraries computing one number,
  speed is the whole comparison, and that one is not close.
- **Word order is where the metrics part company.** An n-gram multiset barely
  notices a reordering (37/40); `fuzzball`'s edit-based ratio drops to 20/40
  first-place, and Fuse to 23/40, because moving a word moves everything after
  it. uFuzzy recovers it only by permuting terms, which is what its `outOfOrder`
  parameter does and what the 5.7 ms/query in that configuration pays for.
- **We lose the prefix query, and so does the metric.** On half a phrase our
  hit@1 is 14/40 against Fuse's 21/40 — and the three Dice arms agree to the
  case, so this is n-gram overlap being the wrong tool rather than an
  implementation fault. Two remembered words out of four match many four-word
  entries about equally, and a position-aware matcher that favours the start of
  a string is right to prefer the one that begins with them. For typeahead over
  prefixes, reach for something positional; for a misremembered whole string,
  reach for this.

The uFuzzy rows are the fairness point of the whole section. At its defaults it
is the fastest thing in the timing table and **finds nothing at all** on three of
the five damage classes. Configured to tolerate one error per term it matches us
on every character-level class — and costs 1.28× our index at 100,000 choices
and 4.2× at 1,000,000, having built no structure to amortize.

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

| Workload                             | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ------------------------------------ | -------------: | ---------------: | ------------------------- |
| `ratio`, 200 sentence pairs          |        42.0 µs |          47.9 µs | ✅ **1.14× faster**       |
| `partialRatio`, 200 sentence pairs   |         884 µs |           413 µs | ❌ 2.14× slower           |
| `tokenSortRatio`, 200 sentence pairs |         341 µs |           230 µs | ❌ 1.48× slower           |
| `tokenSetRatio`, 200 sentence pairs  |         548 µs |           286 µs | ❌ 1.91× slower           |
| `weightedRatio`, 200 sentence pairs  |         790 µs |           404 µs | ❌ 1.96× slower           |

RapidFuzz's `QRatio` has no spelling in this API and is not measured. It is
upstream's own `fuzz.ratio` with one difference: two empty strings score `0`
rather than `100`. Its processor is opt-in and defaults to none, so no
normalization is involved — the scorer it delegates to is the first row above.

### Search and batch scoring

| Workload                                              | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ----------------------------------------------------- | -------------: | ---------------: | ------------------------- |
| `bestMatch` + `ratio`, 20 queries × 2,000 choices     |        2.99 ms |          1.54 ms | ❌ 1.95× slower           |
| `bestMatch` + `tokenSortRatio`, 20 × 2,000 raw titles |        31.6 ms |          19.3 ms | ❌ 1.64× slower           |
| `Matcher` + `tokenSortRatio`, 20 × 2,000 titles       |        5.67 ms |          19.3 ms | ✅ **3.40× faster**       |
| `scoreMatrix` + `ratio`, 50 × 200                     |         642 µs |           235 µs | ❌ 2.74× slower           |
| `scoreMatrix` + `tokenSortRatio`, 50 × 200            |        1.77 ms |          4.30 ms | ✅ **2.43× faster**       |
| `scorePairs` + `ratio`, 200 pairs                     |        44.6 µs |          24.2 µs | ❌ 1.85× slower           |

### Edit operations

| Workload                                        | `rapidfuzz-js` | Python RapidFuzz | Result for `rapidfuzz-js` |
| ----------------------------------------------- | -------------: | ---------------: | ------------------------- |
| Levenshtein editops, 200 pairs × 128 characters |        1.32 ms |           326 µs | ❌ 4.07× slower           |
| LCS editops, 200 pairs × 128 characters         |         593 µs |           139 µs | ❌ 4.27× slower           |

### What the expanded comparison shows

`rapidfuzz-js` wins the smallest Levenshtein workload, `ratio`, the
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
for a plain `ratio` matrix, while `rapidfuzz-js` is 2.43× faster for the
multiword `tokenSortRatio` matrix. Different scorers move different amounts
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

The size ladder uses the same timing loop over a corpus of its own: four-word
phrases from a Zipf-weighted vocabulary at 100, 1,000, 10,000, 100,000 and
1,000,000 choices, each corpus a prefix of the next, and five query classes
from an exact catalog entry to a string sharing nothing with it.

Source files:

- [`bench/comparison/libraries/run.mjs`](bench/comparison/libraries/run.mjs) runs the comparisons.
- [`bench/comparison/indexedSearch/throughput.mjs`](bench/comparison/indexedSearch/throughput.mjs) runs
  the size ladder with the same timing loop, and
  [`indexedSearch/memory.mjs`](bench/comparison/indexedSearch/memory.mjs) measures
  what each structure in it retains, one child process per cell.
- [`bench/comparison/indexedSearch/quality.mjs`](bench/comparison/indexedSearch/quality.mjs)
  measures whether each library finds the entry a damaged query came from.
- [`bench/comparison/indexedSearch/corpus.mjs`](bench/comparison/indexedSearch/corpus.mjs)
  builds the ladder's corpus, shared by all three so the space and hit-rate
  figures describe the strings the time figures used.
- [`bench/comparison/libraries/corpus.mjs`](bench/comparison/libraries/corpus.mjs) builds the corpus.
- [`bench/comparison/shared/timing.mjs`](bench/comparison/shared/timing.mjs) defines the
  JavaScript timing loop.
- [`bench/comparison/libraries/rapidfuzz_bench.py`](bench/comparison/libraries/rapidfuzz_bench.py)
  mirrors that loop for Python.

## Internal performance metrics

The internal suite answers a different question: whether a code change makes
this library faster or slower than its stored baseline. It covers distance
metrics, edit operations, fuzzy scorers, one-shot search, `Matcher`
construction and reuse, matrices, paired scoring, cutoffs, and Unicode cases.

`bench/regression/compare.ts` normalizes every result against control workloads
measured before and after the suite. This reduces distortion from CPU
frequency, background load, and thermal changes.

Important fields in
[`bench/regression/baseline.json`](bench/regression/baseline.json):

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
pnpm bench:quick bench/suites/fuzz.bench.ts
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

The runner writes raw results to `bench/comparison/libraries/last-run.json`, which is
ignored by Git.

The size ladder behind
[Searching a growing collection](#searching-a-growing-collection) is a separate
script over its own corpus:

```sh
node bench/comparison/indexedSearch/throughput.mjs
node bench/comparison/indexedSearch/throughput.mjs --gram=3
node bench/comparison/indexedSearch/throughput.mjs --max=1000000
```

A full ladder including the 1,000,000-choice rung takes about three minutes.
Both matchers, a one-shot `search` and five other packages are timed on every
rung; the four arms that compute Dice are checked against each other before
anything is measured.

What each structure retains is a second script, because every cell needs its own
process:

```sh
node bench/comparison/indexedSearch/memory.mjs
node bench/comparison/indexedSearch/memory.mjs --max=1000000
```

It spawns one child per structure and size, and takes about twelve seconds. The
children are given an 8 GB old space, which the largest arms need — a prepared
1,000,000-choice collection does not fit in a default one.

Whether each library finds the entry the query came from is a third:

```sh
node bench/comparison/indexedSearch/quality.mjs
```

It is deterministic and takes about a minute. uFuzzy and Fuse each appear twice
in its output, at their defaults and configured for typo tolerance, because the
first of those two configurations answers a materially different question.

To include Python RapidFuzz:

```sh
python3 -m venv .venv
.venv/bin/pip install rapidfuzz numpy
node bench/comparison/libraries/run.mjs --python=.venv/bin/python
```

You can also set `RAPIDFUZZ_PYTHON` to the Python interpreter path.

## Interpreting your own run

- Compare ratios, not raw times, across machines.
- Treat differences inside the reported noise band as unchanged.
- Rerun results marked with high spread or machine drift.
- Compare scoring libraries semantically before comparing their speed.
- Benchmark representative string lengths, similarity levels, cutoffs, and
  query-to-choice ratios for your application.
