# Inverted n-gram index — experiment log

Branch `experiment/ngram-inverted-index`. Everything lives in `bench/`; nothing
has been added to `src/`, and nothing should be until this log says the design
is settled.

## The question

> Can **one corpus-wide inverted index** replace the N prepared `NGramProfile`
> tries a Dice/Cosine `Matcher` retains, and still reproduce its results
> exactly — faster, and with less retained memory?

Not "can trigrams find candidates cheaply". The strong form, because it decides
whether this is an option flag or a different corpus representation.

## Where it stands

**Answer so far: yes, except on degenerate alphabets.** Exact on 5,760 fixed
cases and 20,000 randomised corpora; hundreds to thousands of times faster on
realistic text; 8–42x less memory; builds no slower than the Matcher it would
replace. Loses on a 2-letter alphabet and that is inherent.

## Files

| File                                  | What it is                                    |
| ------------------------------------- | --------------------------------------------- |
| `bench/tooling/ngramIndex.ts`         | the prototype                                 |
| `bench/tooling/ngram-index-scale.ts`  | entry point; bundles the payload with esbuild |
| `bench/tooling/ngram-index-report.ts` | parity, counters, memory, sweeps              |
| `bench/ngramIndex.bench.ts`           | the harness-sampled timings                   |
| `bench/comparison/ngram-index.mjs`    | against uFuzzy                                |

```sh
node --expose-gc bench/tooling/ngram-index-scale.ts --parity --runs=3000
node --expose-gc bench/tooling/ngram-index-scale.ts --counters --max=100000
node --expose-gc bench/tooling/ngram-index-scale.ts --counters --sweep --n=10000
node --expose-gc bench/tooling/ngram-index-scale.ts --memory --n=100000 \
  --corpus=zipf-words --gram=3 --arm=index
pnpm bench ngramIndex
node bench/comparison/ngram-index.mjs
```

Flags that matter: `--build=direct|profile`, `--keys=packed|string`,
`--threshold=`, `--sweep`, `--arm=index|profiles`.

## Landed, with the number that justified it

| Change                                                                                      | Measured                                                                                    |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Inverted postings carrying per-gram counts — the index _is_ the scorer, no kernel call left | 26-letter 226–1783x, Zipf 11.7–4509x faster than exhaustive at 100k                         |
| Retain only postings + `gramCount` + `squaredNorm` + gramless elements                      | 80–289 B/choice against 794–8,977 B/choice — 7.6–42x smaller, flat in N                     |
| Bounded insertion-sorted top-k instead of collect-all-then-sort                             | alphabet-2 at 100k: 27ms → 4.7ms                                                            |
| Prefix filtering (skip grams a candidate cannot qualify through)                            | vs full accumulation at 0.8: 1.13–8.74x (26-letter), 1.39–29.83x (Zipf); at 0.95 up to 167x |
| Completion chosen per query between probing and walking the skipped lists                   | removed a 45% regression where probes exploded                                              |
| …chosen on **survivors**, not touched candidates                                            | deciding on touched had the walk win everywhere and cost 26-letter 5x                       |
| Packed integer gram keys, with a reversible downgrade for astral text                       | build 1.21–1.51x faster; query and memory unchanged                                         |
| `addSequence` — extract grams directly, no profile per choice                               | build 1.04–1.95x faster; index now builds **faster than the Matcher** in 5 of 6 shapes      |
| `postingStatistics` — predict the win before any query                                      | see below                                                                                   |
| CSR postings: one `offsets`/`ids`/`counts` triple, not two typed arrays per gram            | **1.29–2.00x less memory**; query 0.99–1.12x; build unchanged                               |
| Count words narrowed to the width that holds the largest frequency                          | included above — `Uint8` on every corpus measured, so 4 bytes per entry became 1            |
| Zero-limit guard, and choices checked to arrive in id order on the way in                   | `limit: 0` indexed `top[-1]` and crashed                                                    |

## Measured and _not_ adopted

- **Rare-gram-first ordering on its own.** Visiting the same posting lists in a
  different order visits the same entries. Only skipping helps, which is what
  prefix filtering does.
- **Precomputed inverse Cosine norms.** Would change operation order and move
  results by an ULP, which moves thresholds, ordering and ties. The exactness
  objective outranks the multiply.
- **Packed keys as a memory optimization.** They are a build optimization: the
  posting arrays outweigh seventeen thousand key strings, so memory moved 1%.
- **An implicit "every count is 1" posting.** The flag never fires: one gram
  repeated in one choice disables it for the whole corpus, and every corpus
  measured had some. `countsWidth` reports 1 byte everywhere, so the adaptive
  width below captures the same saving without the fragility.
- **Term-frequency weighting in the predictor.** Better motivated — a query
  draws grams by term frequency, not document frequency — but on these corpora
  the two agree to three decimals and order the configurations identically, one
  inversion each. Kept as `termWeightedShare` for corpora with heavy
  intra-document repetition, which none of these have.
- **Prefix ordering by `length / queryCount`.** Adopted, but it is not a
  measured win here: n-gram query counts are almost always 1, so it picks the
  same order. It is the correct rule when they are not, and it is free.

## The predictor for `index: 'auto'`

`weightedShare = Σ len² / Σ len / N` — the share of the corpus covered by the
gram a query is _likely_ to ask for. Build-time, one pass, no queries.

```
0.0013  →  292x faster     26-letter, trigrams
0.0335  →    9.6x          26-letter, bigrams
0.1141  →   11.4x          Zipf words, trigrams
0.2067  →    1.4x          10-letter, bigrams
------- crossover, somewhere in here -------
0.2945  →    0.97x         4-letter, trigrams
0.6115  →    0.61x         5-letter, bigrams
0.9964  →    0.01x         2-letter, bigrams
```

Cutoff of **0.2** is safe on everything measured. After CSR the index got
faster, and the crossover moved out with it — the last win is now 0.304 and the
first loss 0.478 — so 0.2 has more headroom than it did, not less. The alphabet sweep exists
because the first round measured 2 and 26 only, leaving the middle — where the
answer changes — unmeasured.

## The run

`pnpm bench ngramIndex`, trigrams, three arms kept apart — **A** exhaustive
prepared Matcher, **B** simple inverted, **C** prefix-filtered.

```
                                          A            B          C
1,000 choices, 100 queries, 0.5      49.22ms      0.431ms          —
10,000 choices, 100 queries, 0.5     644.5ms      0.711ms          —
10,000 choices, 100 misses, 0.5      652.6ms      0.743ms          —
10,000 sentences, 100 queries, 0.5   642.0ms      0.945ms          —
10,000 choices, cosine, 0.5          809.8ms      0.998ms          —
10,000 sentences, 100 queries, 0.8         —      0.959ms    0.808ms
100,000 choices, 1 query, 0.5        59.93ms     0.0313ms          —
100,000 choices, 1 query, 0.8              —     0.0313ms   0.0158ms
10,000 best(), 100 exact hits         3.464ms     0.698ms          —

build, 10,000 choices                 39.23ms      34.82ms    96.50ms via profiles
```

So: **B is where the win is** — three orders of magnitude at 10k, and it is the
simple representation. **C adds up to 2x on top** where the threshold is high
and the corpus has skew, and nothing at all where it does not. Attributing that
separately is the reason the arms are kept apart.

`best()` with no threshold is the narrowest margin at 4.96x, because the
exhaustive scan breaks out the moment it scores 1.

Memory, trigrams at 100k: **14.0 MB against 898 MB** (26-letter, 64x), **13.1 MB
against 875 MB** (Zipf, 67x).

## Where it loses, and why

- **2-letter alphabet**: 8 possible trigrams, so every posting list is the whole
  corpus and no prefix can be short. 0.1–0.2x of exhaustive on an exact hit at
  every threshold. Inherent.
- **`best()` with no threshold**: 4.96x, the narrowest margin, because the
  exhaustive scan breaks out the moment it scores a perfect match and the index
  has no such exit.
- **A cheap metric would narrow all of this.** uFuzzy scans every candidate and
  still beats our exhaustive path 20x, so the headline gap was partly Dice being
  expensive per candidate, not the scan being slow.

## Against uFuzzy

Not a correctness comparison — subsequence-with-position against position-free
n-gram overlap. At 100k, per query: uFuzzy 1.9–3.3ms whatever is asked of it
(a linear scan, and it reads like one); ours 0.002ms on a selective query,
converging on uFuzzy where the query's grams are common. uFuzzy builds nothing,
where we pay a build. The match counts say the rest: a single common word finds
48,298 items in uFuzzy and **none** under Dice at 0.5.

**Conclusion:** where the job is "items containing what I typed", Dice is the
wrong measure and no index rescues it. Where it is comparing strings of similar
length — dedup, record linkage, titles against a catalogue — the index is what
makes it scale.

## Correctness

Parity compares against `matcher.search`/`.best` on **key, score and order**:
5,760 fixed cases × 4 build/key combinations, plus 20,000 randomised corpora.
Covers gram sizes 2 and 3, thresholds `null`/0/0.5/0.8/1, limits 0/1/3/`null`,
duplicate choices, sequences shorter than `gramSize`, gramless queries, astral
characters and lone surrogates.

Mutation-verified twice — the suite is only worth what it catches:

- `min` → `max` in the Dice accumulator: caught, with a counterexample.
- a prefix two grams too short: caught, drops an exact match.

Fixed along the way: choices must arrive in id order (checked on the way in, not
at `compact`, where a duplicate would already have written itself into every
list); `limit: 0` indexed `top[-1]` and crashed — the parity suite had no zero in
its limit set, which is why it never showed up.

## Scope, stated plainly

Gram elements are integers — code points, in practice. Both `add` and
`addSequence` reach `integerElement`, so this is _"can an integer/code-point
n-gram index replace prepared profiles for ordinary text"_. The metric itself is
more general: its trie is keyed by `unknown` and treats `NaN` as unmatchable. An
index for that would intern arbitrary elements to integer symbols first, and
that is not Stage B.

## Open

1. **Direct query-trie traversal** for plain Dice/Cosine accumulation, skipping
   the flattened arrays entirely. Deliberately postponed: the scratch reuse is
   in, and further query-preparation micro-work would answer a smaller question
   than the one still open.
2. **Sparse counts.** 93.5% of posting lists are all-ones (59.8% on Zipf) but a
   corpus maximum of 3 forces a counts array anyway. A per-list flag would fire
   where the corpus-wide one cannot — only worth it if one byte per entry stops
   being cheap enough.
3. **Delta-encoded ids.** Postings are sorted, so the gaps are small — but it
   costs the binary search, so only after the cheaper layout wins are banked.
4. **`best()` bootstrap** — score the rarest gram's candidates first and use that
   as a cutoff, to get the early exit the exhaustive path has.
5. **Cosine has no prefix filtering**; its threshold does not become a
   shared-count bound without the norms. WAND/MaxScore is the shape that fits,
   and belongs in Stage C only if plain inverted Cosine is not fast enough.
6. **Updates are rebuild-only.**
7. **The `src/` shape**: a `MetricCompilation` capability, because
   `tests/architecture/imports.test.ts:40` forbids `search/` from importing an
   algorithm.

## Benchmark hygiene

Three arms are kept separate on purpose, so the win can be attributed rather
than just claimed: **A** exhaustive prepared Matcher, **B** simple inverted full
accumulation, **C** optimised inverted (prefix filtering).

- Timings that get quoted come from `pnpm bench ngramIndex`, not the scale
  script — the latter has warm-up but no adaptive sampling.
- The bench file carries a contamination control duplicating a case from
  `bench/ngram.bench.ts`. It read 35.30ms against that file's 34.03ms, so the
  in-file comparisons are trustworthy.
- Memory arms run **one per process**. Measured in one process the deltas came
  out negative: the baseline was taken over another arm's garbage.
- Corpora are built with `Array.join`, never `+=`. A string concatenated
  character by character is a chain of cons strings that the first
  `convSequence` flattens, so the corpus _shrinks_ mid-measurement.
