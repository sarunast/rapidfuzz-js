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

**Stage B is answered.** The prototype was frozen and then reopened once, for
dense postings — see below — because the real corpus showed the remaining weak
case was a representation problem rather than a constant factor.

> A corpus-wide inverted n-gram representation can replace per-choice prepared
> profiles for integer/code-point Dice and Cosine search while preserving exact
> `Matcher` results — key, score and order.
>
> Full inverted accumulation is on its own orders of magnitude faster than
> exhaustive search on selective trigram corpora, and 17–33x faster on a real
> one. Prefix filtering adds to that for Dice at high thresholds and skewed
> distributions; **it is not part of the architectural case.**
>
> Retained memory falls by 52–64x — 4.8 MB against 250 MB on the real corpus.
> This is not the usual index trade of memory for speed; it is less of both.
>
> Performance tracks posting selectivity. Dense low-alphabet corpora remain the
> adverse case, and no amount of filtering rescues them.

**The index is the product. Prefix filtering is an optimisation.** That
separation is the most useful thing this round produced, and the internal shape
keeps it: `NGramIndex` accumulates exact Dice and exact Cosine, and carries an
optional Dice prefix strategy beside them.

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
node --expose-gc bench/tooling/ngram-index-scale.ts --peak --n=1000000 \
  --corpus=zipf-words --gram=3 --build=direct
pnpm bench ngramIndex
node bench/comparison/ngram-index.mjs
```

Flags that matter: `--build=direct|profile`, `--keys=auto|bmp|full|string`,
`--dense-cutoff=<share>|off`, `--threshold=`, `--sweep`,
`--arm=index|profiles|matcher`, `--corpus=`, `--n=`. `--dense` runs the probe
that decided dense postings were worth building.

Every row records `buildMode`, `keyMode`, `threshold` and `limit`. Two runs
differing only in `--keys` used to emit byte-identical rows, which made a
directory of JSON output unattributable after the fact.

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

## Dense postings — store who does _not_ have the gram

The real corpus exposed the case the CSR layout was not built for. `node_modules`
is in almost every path, so its trigrams' posting lists name nearly the whole
corpus, and `common substring` — the query made of exactly those — was the one
shape where the index barely beat the Matcher at 2x.

For a list covering most of the corpus the natural representation is the
complement. A **dense** list stores exceptions to a default frequency of 1: an
absence at count `0`, a repeat at `2` or more. Dice takes `min(q, 1)` for the
whole corpus in one addition and then walks only the exceptions; Cosine takes
`q`, and an exception contributes `q × (count − 1)`. Both stay exact.

**Probed before it was built**, because the whole idea rests on a number nothing
had measured: what share of the work a query does lands in lists dense enough to
invert. On the file-path corpus, **17 lists of 11,189** — 0.15% of them — and
they carry a third of the traffic:

| query            | sparse work | hybrid work | ratio |
| ---------------- | ----------: | ----------: | ----: |
| exact hit        |     288,791 |      86,841 | 3.33x |
| common substring |     142,143 |      17,296 | 8.22x |
| rare substring   |       7,238 |       7,238 | 1.00x |

Both sides carry their selection scan, and they are not the same scan: the sparse
path visits what it touched, a dense list forces every candidate. Charging that
to the hybrid alone made a corpus where `touched` is already `N` look like a
regression when it is a small win.

Measured after, `--dense-cutoff=off` against the default, same process:

| corpus, query                | sparse |  dense |      ratio |
| ---------------------------- | -----: | -----: | ---------: |
| file-paths, exact hit (dice) | 0.8418 | 0.4384 |      1.92x |
| file-paths, 2 typos (dice)   | 0.8295 | 0.3891 |      2.13x |
| file-paths, 2 typos (cosine) | 0.8316 | 0.3613 |      2.30x |
| file-paths, common substring | 0.1070 | 0.1042 |      1.03x |
| file-paths, rare substring   | 0.0903 | 0.0926 |      0.98x |
| alphabet-2 100k, all classes |      — |      — | 1.08–1.24x |

**1.83–2.30x on the classes that dominate**, 1.08–1.24x on the adverse
two-letter corpus, and exactly neutral where no list qualifies — `zipf-words` and
`alphabet-26` produce none at all, so the flag array is `null` and nothing reads
it. Retained memory **4.77 MB → 3.81 MB, 20% less**. Build costs 6% more
(103 ms → 109 ms) for the merge that computes each complement.

**The most useful number is the one that did not move.** `common substring` shed
30x its posting traffic — 129,218 entries to 4,349 — and came out 1.03x. Its cost
was never accumulation; it is the corpus-wide selection scan, which that query
was already paying because it touched 12,925 of 12,947 candidates. Posting
traffic is not the only budget, and this is where the next win has to come from.

**The cutoff is 2/3, not 1/2.** Inverting costs a second thing: any query
touching a dense list has to score every candidate. A dense gram changes
accumulation by `N − 2·length + exceptions` and selection by at most `N − length`,
and the sum only turns negative above `2N/3`. At exactly one half the storage
saving is zero and the scan is pure loss.

Prefix filtering **falls back to full accumulation** when a query reaches a dense
list. Its bound is stated over lists that name who _has_ a gram, so a dense list
inverts the meaning of every step — and a dense list is the cheapest thing in the
index, one addition plus its exceptions, so there was never anything to gain by
skipping one.

Three bugs, all caught by parity, none by inspection:

- **The invariant broke, correctly.** "Score `> 0` ⇔ touched" is a property of
  the sparse representation; a dense list hands every candidate a base frequency,
  so nothing is untouched. The check now reads a `scannedAllCandidates` counter
  instead of assuming.
- **Duplicate results.** The first version materialised `touched` as every id,
  and the sparse loops then pushed ids that were already there. Removing the
  materialisation fixed it and was faster anyway — a million pushes cost more
  than the loop-invariant branch selection pays instead.
- **A gramless choice scored 1 under Cosine.** It is in no posting list, so the
  sparse path never reached one; scanning every candidate does, and `0/0` clamped
  to a perfect score. Both score functions now answer `0` for a choice with no
  grams.

## Where a query's time goes

`common substring` shed 30x its posting traffic for a 3% gain, so the next thing
to measure was the loop that gain did not reach. Stages are timed as prefixes of
the whole call and the differences read off — timing a stage alone would measure
it on state the previous stage never built — and reported as the minimum of many
runs, because these are tens of microseconds on a machine that spikes.

File paths, 12,947 choices, trigrams, `exact hit`, threshold 0.5, dense on:

| stage        |     ms | share |
| ------------ | -----: | ----: |
| buildProfile | 0.0060 |  1.5% |
| flatten      | 0.0081 |  2.1% |
| accumulate   | 0.2083 | 54.4% |
| select       | 0.1664 | 42.0% |

Preparation is 4%, so **skipping the query's `NGramProfile` cannot matter here**
whatever it saves. Accumulation is still the largest single stage. Selection is
the surprise: it is nearly half of a query that touches 75,805 posting entries.

Inside selection, every variant scanning the same accumulated state:

| variant                        |     ms |
| ------------------------------ | -----: |
| loop only                      | 0.0047 |
| + accumulator read             | 0.0294 |
| + divide, inline               | 0.0220 |
| + divide, through the callback | 0.0323 |
| callback over hoisted locals   | 0.0219 |
| **callback + top-5 insertion** | 0.0447 |
| **inline + top-5 insertion**   | 0.0259 |
| length band, then divide       | 0.0254 |

Two things fall out. **Inlining the score into the loop is 1.7x** — the callback
is not free once it runs per _corpus_ candidate rather than per touched one, and
hoisting the fields it reads off `this` recovers most of that on its own. And the
**length band buys nothing**: `2·min(q,g)/(q+g) ≥ t` rejects almost nobody here,
because a corpus of file paths has file-path-shaped lengths.

The isolated loops are optimistic — they rerun over an accumulator nothing is
rewriting — so the honest reading is that selection is 0.045–0.166 ms of a
0.39 ms query, and the part of it worth attacking is the per-candidate callback.

### Skipping the unmodified candidates: measured, and it does not pay

A dense list gives every candidate a base score, and a candidate the accumulation
never wrote to scores exactly `2·base/(q + g)` — monotonic in `gramCount` alone.
So a precomputed `gramCount` order would let top-k stop after a handful of them
instead of scanning the corpus. Whether that pays depends on one number nobody
had counted: how many scanned candidates are actually modified.

| query            | scored | modified | unmodified | modified share |
| ---------------- | -----: | -------: | ---------: | -------------: |
| exact hit        | 12,947 |   12,110 |        837 |          93.5% |
| 1 typo           | 12,947 |   12,105 |        842 |          93.5% |
| 2 typos          | 12,947 |   12,040 |        907 |          93.0% |
| common substring | 12,947 |    1,793 |     11,154 |          13.8% |
| rare substring   |  3,616 |    3,616 |          0 |         100.0% |

**93% modified on the classes that dominate**, so the ordering would skip 6.5% of
a scan that is itself 43% of the query. It pays on exactly one class — `common
substring`, 86% unmodified — and that is the cheapest query on the corpus at
0.104 ms against the Matcher's 0.205 ms. Not built: the design is sound and the
corpus it needs is not this one.

### The cutoff sweep cannot discriminate

| cutoff | exact hit | 1 typo | common substring | retained |
| ------ | --------: | -----: | ---------------: | -------: |
| off    |    0.8893 | 0.9170 |           0.1118 |  4.77 MB |
| 0.50   |    0.3895 | 0.3875 |           0.1079 |  3.84 MB |
| 0.6667 |    0.3993 | 0.4277 |           0.1127 |  3.83 MB |
| 0.90   |    0.3911 | 0.4246 |           0.1149 |  3.86 MB |

Every cutoff from 0.5 to 0.9 measures the same, in latency and in bytes, because
**this corpus has no posting list in that band** — the 17 that qualify are all
far above 0.9. So the sweep confirms dense against sparse (2.3x) and says nothing
about where the cutoff belongs. `2/3` stays a derivation rather than a
measurement, which is the honest label for it.

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

## The real corpus

Synthetic corpora say where the crossover is. This says which side of it a real
workload sits on. **12,947 file paths** from the checkout's `node_modules`,
sampled at proportional positions across the sorted list — a contiguous prefix is
one or two packages' worth of near-identical paths, a fake corpus wearing a real
one's name, and it overstated posting traffic 19x. An integer stride is not
enough either: at 10,000 of 12,947 it floors to 1, which is the prefix again.

Trigrams, threshold 0.5, limit 5, 8 distinct queries per class, milliseconds:

| query            | exhaustive | indexed | prefix | speedup |
| ---------------- | ---------: | ------: | -----: | ------: |
| exact hit        |       14.1 |  0.8343 | 0.7943 |     17x |
| 1 typo           |       15.5 |  0.7968 | 0.7255 |     20x |
| 2 typos          |       18.1 |  0.8095 | 0.6270 |     22x |
| unrelated        |      1.293 |  0.0073 | 0.0059 |    176x |
| short            |     0.1988 |  0.0039 | 0.0027 |     51x |
| common substring |     0.2048 |  0.0972 | 0.0486 |      2x |
| rare substring   |     0.2009 |  0.0782 | 0.0052 |      3x |

Cosine on the same corpus: 31–33x on the hit and typo classes, 93–1,603x on the
selective ones. Build: **index 98–102 ms, Matcher 120–242 ms** across four runs.

Repeated four times, because this machine spikes: the hit and typo classes read
16–24x every time, with one outlier run at 11–18x where the indexed arm alone
went 1.8x slower — load, not a finding. The table is the median run.

Three things this corpus says that no synthetic one did:

- **Candidate pruning does almost nothing here.** An exact hit touches 12,932 of
  12,947 choices, because `node_modules`, `dist` and `.js` are in nearly every
  path. The win is not fewer candidates; it is that a posting entry is far
  cheaper than a profile walk.
- **17–33x, not 900x.** An order of magnitude below `alphabet-26`, and still
  decisive. `weightedShare` is 0.33 — between the 0.304 that was the last win and
  the 0.478 that was the first loss in the synthetic sweep. Real file paths sit
  _inside_ the crossover band, on the winning side.
- **`common substring` is the adverse class within a real corpus**, at 2x. It is
  the one query shape where a caller would notice the index barely helping.

Memory, same corpus, one arm per process:

| arm             |    bytes | per choice |
| --------------- | -------: | ---------: |
| index           |  4.82 MB |      372 B |
| profiles only   | 250.2 MB |   19,325 B |
| whole `Matcher` | 251.0 MB |   19,385 B |

Two comparisons, deliberately: index against **profiles** is the representation
question, and index against the whole **Matcher** is what a caller retains. They
differ by 0.77 MB — the per-choice row holding item, key and prepared value is
**0.3% of what a Matcher keeps**, so the profiles are 99.7% of it. The earlier
index-vs-Matcher figures were not meaningfully flattering.

## Peak build memory

Retained bytes are the architectural claim; peak is what decides whether a corpus
can be indexed at all. Until `compact()` runs, every posting list is a pair of
growable JS arrays in a `Map`, so the build's high-water mark is nothing like the
CSR arrays it settles into. Sampled inside the build loop — the build is one
synchronous run, so a timer would never get a look. All figures over the corpus
baseline, direct build, trigrams:

| corpus      |         n | retained | peak build | peak RSS | ratio |
| ----------- | --------: | -------: | ---------: | -------: | ----: |
| alphabet-26 | 1,000,000 |  80.5 MB |     647 MB |   962 MB |  8.0x |
| zipf-words  | 1,000,000 |  74.5 MB |     841 MB |  1056 MB | 11.3x |
| file-paths  |    12,947 |  4.68 MB |    56.7 MB |   196 MB | 12.1x |

Peak build is `heapUsed + arrayBuffers` **sampled as one sum**, over the same sum
at the baseline. Tracking the two maxima separately and subtracting a combined
baseline from a heap-only peak is not the peak of anything: it undercounts by
whatever the corpus already held in buffers, and the two need not peak at the
same instant.

**Peak runs 8–12x the final size, and that is fine.** A million choices index
inside ~1 GB RSS, while a million prepared profiles would retain ~3.3 GB — so the
builder's high-water mark is **roughly three times below** (3.15x against the
Zipf RSS, 3.45x against the 26-letter one) the steady state of the representation
it replaces. Peak against peak would be a kinder comparison and is not the one
that matters; RSS is what a machine has to find. So the two-pass CSR build (count
posting lengths, allocate exactly, fill) stays a named fix and is **not**
implemented: it trades a second corpus traversal for a problem nothing has yet.

## Correctness

Parity compares against `matcher.search`/`.best` on **key, score and order**:
**80,640 fixed cases across the whole build × key × dense product**, generated rather
than listed — a hand-written list claimed to be the product while missing
`profile + bmp` and `direct + full` — plus randomised corpora that draw their
configuration too. Covers gram sizes 2 and 3, thresholds
`null`/0/0.5/0.8/1, limits 0/1/3/`null`, duplicate choices, sequences shorter
than `gramSize`, gramless queries, astral characters and lone surrogates.

`addSequence` and `add` are two separate builders and each key scheme is a
separate keying path, so parity covers the product rather than whichever
configuration the last flag selected. Every measuring mode now runs a **smoke
subset first, under the configuration it is about to measure** — milliseconds
against a run of minutes, and the alternative was producing timings for an
unvalidated path. The memory and peak modes included: they are not scoring
benchmarks, but they build through a specific builder and key scheme, and a byte
count for a representation that answers wrong is no better than a timing for one.

Mutation-verified twice — the suite is only worth what it catches:

- `min` → `max` in the Dice accumulator: caught, with a counterexample.
- a prefix two grams too short: caught, drops an exact match.

Fixed along the way: choices must arrive in id order (checked on the way in, not
at `compact`, where a duplicate would already have written itself into every
list); `limit: 0` indexed `top[-1]` and crashed — the parity suite had no zero in
its limit set, which is why it never showed up.

Two more the ladder brought with it:

- **`radixFor` answered a rung for a negative element.** `-1 < 256` is true, so
  it named the rung that had just failed, `rekey` did nothing, and the build died
  with "key scheme failed to widen" — on an element the joined-string scheme
  represents exactly. Negatives now go straight to strings. Unreachable from
  `addSequence`, which sees only code points; reachable from `add`.
- **A pinned rung too wide for the depth** overflowed `partial * radix + value`
  past the safe-integer range, and lost precision shows up as two grams sharing a
  key — a wrong score, not an exception. The constructor refuses it, which is how
  `--keys=full --gram=3` stopped being a silent request: the full code-point
  radix reaches two elements, not three.

## Scope, stated plainly

Gram elements are integers — code points, in practice. Both `add` and
`addSequence` reach `integerElement`, so this is _"can an integer/code-point
n-gram index replace prepared profiles for ordinary text"_. The metric itself is
more general: its trie is keyed by `unknown` and treats `NaN` as unmatchable. An
index for that would intern arbitrary elements to integer symbols first, and
that is not Stage B.

## Open — and closed to further optimisation

**Stage C is architecture, not another 20% off postings.** The question is how an
index integrates with `Matcher` without damaging the scorer API, and the leading
answer is an **explicit** `createIndexedMatcher` rather than silently changing
what `createMatcher` builds — indexing changes construction strategy, rebuild
expectations, which scorer families are supported, which input representations
are, and what is retained. Automatic selection can come later, once the explicit
form has proved itself; the `weightedShare` predictor below is what it would use.

Two production-hardening items belong to that stage, not this one:

- **Explicit width guards.** `gramCount`, `ids` and `offsets` are `Uint32Array`,
  so `choiceCount`, total posting entries and a profile's `gramCount` all carry a
  32-bit bound. Nothing realistic approaches them, and a library should throw
  rather than truncate into a typed array.
- **The two-pass CSR build**, if peak build memory ever becomes the limit. It is
  measured above and it is not the limit today.

Everything below stays deliberately unbuilt:

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
- **Each query class carries 8 distinct queries**, cycled through the timed runs.
  Timing one query in a loop rewarms exactly the posting slices that query
  touches — free for the index, worthless to the exhaustive arm, which walks the
  whole corpus either way.
- **p95 is `null` below 40 samples**, which means at 10k and above. With 5
  samples `sorted[floor(0.95 × 5)]` is the maximum, reported under a percentile's
  name; with 15 it is still the maximum. The tail number for a large corpus
  belongs to `pnpm bench ngramIndex`, where adaptive sampling takes hundreds of
  samples, not to a script whose exhaustive arm costs 16 ms a call.
- **`substitute` samples positions without replacement.** Picking each
  independently let two edits land on the same character, and on a small
  alphabet the second could undo the first — a "2 typos" row measuring an exact
  hit.
- **No defensive copy in front of `createMatcher`.** It takes a
  `readonly TItem[]`; a clone charged the Matcher's build for something no real
  caller does.
- **`--n` names a size rather than filtering the ladder.** `--n=50000` used to be
  accepted, match nothing, and print an empty run that looked like a finished
  one. A run that measures nothing now throws, and `--gram` accepts only 2 or 3.
- **`--keys` is one value.** Two flags writing to separate variables let
  `--keys=bmp --keys=string` produce a string-keyed index carrying a pinned BMP
  rung — a state nothing asked for and no row recorded.
