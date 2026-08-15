# `algorithms/ngram/`

An **algorithm foundation**, not a metric family. Nothing here is exported from
a public entry point; `algorithms/dice/` and `algorithms/cosine/` are the only
consumers, and it depends on no public algorithm. That is what its position
states — it sits beside the public algorithms because it is below them, in the
same layer as `algorithms/bitmask/` and `algorithms/affix.ts`, and
`tests/architecture/imports.test.ts` fails an edge from here into any of the
twelve published directories.

Both metrics reduce to one question — _how many grams do these two sequences
share?_ — differing only in how a shared gram is scored:

```text
Dice    2 · Σ min(aᵍ, bᵍ) / (gramsA + gramsB)     →  0..1
Cosine  Σ aᵍ · bᵍ / √(‖a‖² ‖b‖²)                  →  0..1
```

Everything here exists to answer that numerator quickly, at three different
lifetimes, over element types ranging from Latin-1 characters to arbitrary
objects.

There is no barrel. Import the file you need, as every `algorithms/bitmask/`
consumer already does.

---

## Architecture

### Lifetimes, which is what the layout is organised by

The same two quantities are computed by three different engines. Which one runs
is decided by the caller's shape, not by the metric:

| lifetime                       | entry        | who                       | amortises                                                     |
| ------------------------------ | ------------ | ------------------------- | ------------------------------------------------------------- |
| **one comparison**             | `compare.ts` | `diceSimilarity(a, b)`    | nothing — there is no later query to pay a profile forward to |
| **one query, many candidates** | `kernel.ts`  | `search`, `createMatcher` | the query side, compiled once at preparation                  |
| **one corpus, many queries**   | `inverted/`  | `createIndexedMatcher`    | the _choices_, inverted once at build                         |

That progression is the whole design. `compare.ts` and `kernel.ts` compute
identical numbers and are two files because their lifetimes differ: a direct
call that flattened its query into indexed arrays would pay for a loop that
never runs, and a search that walked both tries per candidate would re-walk one
side thousands of times.

### Module graph

Every intra-subsystem edge, in topological order — each module may import only
what is listed for it, and nothing below it:

```text
key                (leaf)
gramSize           (leaf, read by dice/ and cosine/ only)
packing            → key
profile            → key, packing
compare            → key, packing, profile
kernel             → key, packing, profile, compare
─────────────── the index is built on the above, never the reverse ───────────────
inverted/keys      → key
inverted/builder   → key, inverted/keys
inverted/query     → inverted/builder
inverted/dice      → inverted/keys, inverted/builder, inverted/query
inverted/cosine    → inverted/keys, inverted/builder, inverted/query
```

Outward, the subsystem reaches only `core/` — `core/sequence` for `convSequence`
and `elementsEqual`, `core/types` for `Sequence`, and `core/scoring/choiceIndex`
for the `ChoiceIndex`/`ChoiceIndexBuilder` protocol the index implements. It
imports no algorithm, and nothing imports it except `algorithms/dice/` and
`algorithms/cosine/`.

Three rules, all enforced by `tests/architecture/imports.test.ts`:

1. **`profile.ts`, `compare.ts` and `kernel.ts` never import `inverted/`.** The
   index is an optional acceleration strategy built _on_ n-gram semantics, not
   the foundation.
2. **`inverted/` reaches back into `ngram/` for `key.ts` and nothing else.** The
   index shares an _encoding_ with the profiles without sharing a
   _representation_: it never sees a `GramNode`, an `NGramProfile`, or a
   comparison. The two halves meet only at the radix ladder.
3. **`key.ts` and `gramSize.ts` are leaves.** One is integer arithmetic over a
   radix ladder, the other reads a single option; an edge out of either is the
   first sign policy has leaked in.

The test also pins both directory listings, so a new file here cannot appear
without someone acknowledging it.

### What each module owns

| file                  | owns                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `key.ts`              | radix-ladder arithmetic: `feasibleRadices`, `canonicalRadix`, `packGram`, `unpackGram`.                          |
| `packing.ts`          | element → digit → packed key, and the element domain a key is spelled in.                                        |
| `gramSize.ts`         | the `gramSize` option: validation and parsing.                                                                   |
| `profile.ts`          | what an n-gram profile _is_ and how one is built — both storages, the trie node shape, the fixed-depth builders. |
| `compare.ts`          | one comparison: `sharedFrequency`, `dotProduct`, and the direct counter that skips profiles entirely.            |
| `kernel.ts`           | one query, many comparisons: the query compiled once, then run per candidate.                                    |
| `inverted/keys.ts`    | the index's _adaptive_ key policy — narrowest radix, widened on demand, joined strings as the floor.             |
| `inverted/builder.ts` | corpus ingestion and the CSR posting store. Generic across both metrics.                                         |
| `inverted/query.ts`   | query scratch, zero-filling, selection and ranking.                                                              |
| `inverted/dice.ts`    | the Dice scoring engine, `Int32Array` accumulator.                                                               |
| `inverted/cosine.ts`  | the Cosine scoring engine, `Float64Array` accumulator.                                                           |

---

## Implementation

### Key encoding

A gram is packed positionally into a single number, most-significant element
first — `packGram` and its exact inverse `unpackGram`. Reversibility is
load-bearing: it is what lets a packed profile answer a trie one, and what lets
an index that has ingested a million choices change radix without re-reading
any of them.

The ladder has three rungs, narrowest first:

```text
0x100      Latin-1     'abc' → 0x616263, 24 bits
0x1_0000   BMP                 the same three letters cost 48
0x11_0000  full code points
```

A rung is feasible at a depth when `radix ^ gramSize` stays inside
`Number.MAX_SAFE_INTEGER`. Depth is therefore what decides reach — a byte radix
holds six elements, a BMP radix three, the full range two:

| `gramSize` | `canonicalRadix`               |
| ---------- | ------------------------------ |
| 1–2        | `0x11_0000`                    |
| 3          | `0x1_0000`                     |
| 4–6        | `0x100`                        |
| 7+         | `null` — no packed rung exists |

Small integer keys are also the ones a `Map` handles best, which is why the
narrow rungs are worth having at all.

**The two sides pick opposite ends of the ladder, deliberately.** A prepared
profile takes the _widest_ feasible rung, because two profiles meet with no
shared context to re-key against and one canonical radix per depth is what lets
them compare keys at all — the same gram at two radices is two different
numbers. An index takes the _narrowest_, because it owns every choice it holds
and can widen them all when a build demands it. That is the entire reason
`inverted/keys.ts` exists beside `key.ts`.

### Element domains

`'a' !== 97` at this layer, and a packed key of `97` could mean either, so the
domain (`'number'` or `'char'`) travels with the keys. Two profiles that packed
different domains share nothing by definition and are answered `0` without a
walk.

A sequence's domain comes from its first element and every later one must agree.
`[97, 'b']` is packable twice over on its own terms, and packing it would make
`'b'` and `98` the same gram where a trie keeps them apart — so a mixed sequence
refuses packing and stays a trie.

### Profiles: two storages

|         | packed                                                | trie                                                                  |
| ------- | ----------------------------------------------------- | --------------------------------------------------------------------- |
| shape   | sorted distinct keys + counts, two typed arrays       | nested `Map`s, depth `gramSize`                                       |
| accepts | integers/chars inside the canonical radix, one domain | anything, compared by identity                                        |
| cost    | ~12 bytes a gram                                      | ~228 bytes a gram (127 `Map` objects for 86 grams on an 89-char path) |

Packing is worth roughly **19x** the memory, so it is the default; the trie is
the fallback for object elements, negatives, `NaN`, mixed domains, and astral
code points at trigram depth. At 100k retained prepared bigram profiles the node
shape alone — `children` and `counts` nullable, never both populated, neither
allocated until something is inserted — is 331 MiB against 750 MiB for the
obvious shape that gives every leaf an empty `Map`.

`ProfileStorage` is a discriminated union rather than a nullable `packed` beside
a nullable `root`, which would spell two states that mean nothing.

### Profile construction

Depths 2 and 3 get literal builders that carry the window forward: consecutive
grams overlap in every element but one, so a rolling form reads each element
once where the generic loop re-reads it `gramSize` times. Worth **1.05–1.15x**
on a direct comparison, growing with length. The packed key path has the same
specialisation and needs no digit array at all — **1.06–1.55x**, most of it at
short inputs where the dropped allocation _is_ the call.

Three details that are easy to break:

- **`NaN` is unmatchable.** `NaN !== NaN` is the element equality every other
  metric here uses, while a `Map` keyed by one matches it under SameValueZero. A
  gram holding `NaN` is therefore never inserted — but it still counts toward
  `gramCount` and `squaredNorm`, so denominators and search bounds stay right.
  Tracked by comparing each window's start against the last unmatchable index
  seen, so every element is tested once rather than once per window.
- **`squaredNorm` is maintained incrementally** as `(c + 1)² − c² = 2c + 1` per
  occurrence, rather than by a second pass over the finished structure.
- **A sequence with no grams keeps its elements**, and only such a profile does
  — so the retained sequence is itself the "compare these directly" signal that
  `zeroGramSimilarity` reads. A profile that has grams would otherwise hold its
  converted input alive for nothing.

`packedProfile` sorts _every_ gram and then counts runs, rather than tallying
into a `Map` and sorting only the distinct keys. The `Map` shape wins exactly
where a long sequence draws on a tiny alphabet (0.64x on 4096 characters of
bigrams over 26 letters) and loses **1.56–2.63x** everywhere else, including the
shapes prepared search is made of. The sort is 87% of the build — 0.208 ms of
0.239 ms at 4096 characters.

### Comparison

`combine` reduces the four representation pairs to three cases by normalising
orientation — both operations are symmetric, so a trie-first mixed pair is the
packed-first pair with its operands swapped:

```text
packed × packed   →  packedIntersect       (domains must agree, else 0)
packed × trie     →  packedAgainstTrie     (decode each key, look it up)
trie   × trie     →  sharedFrequencyTries / dotProductTries
```

Different depths share no gram, and packing is where that stops being
structural: a depth-1 trie could never line up with a depth-2 trie, while the
unigram `[97]` and the bigram `[0, 97]` both key to `97`. `combine` tests for
it; the prepared kernels do not, because a scorer only compares profiles it
prepared itself.

**`packedIntersect` has two arms.** Both sides are sorted, so the default is a
linear merge — but when one side is at least `PROBE_LENGTH_RATIO` (8) times
longer, it binary-searches into the long side instead. That constant was swept
over query lengths 5–50 against length ratios 1–64: a probe is 0.13–0.44x of a
merge from ratio 8 upward and inside its noise below, while ratios 1–4 are where
the merge is up to 1.15x ahead. The corner it exists for is a short query against
long choices, where a merge measured **3.0x slower** than the trie kernel it
replaced on 5 grams into 500. Because the driving side ascends, each probe starts
where the last one landed.

The midpoint is `low + ((high - low) >>> 1)`, not `Math.floor((low + high) / 2)`.
A sequence may hold up to `0xffff_ffff` elements, so `low + high` can pass 2³² and
wrap to a midpoint outside the window; `high - low` is bounded by the length and
`>>>` is exact over it. It is also **4.3x faster** than the `Math.floor` form over
5 × 500 keys — the arithmetic _is_ the loop.

**`directSharedFrequency`** skips profiles entirely for Dice. A prepared profile
sorts its grams so later queries can merge against it; a direct comparison has
no later query, so sorting is `O(n log n)` spent ordering something read once.
Instead it tallies the smaller side into one `Map` and spends the larger against
those counts — `O(n + m)`. Decrementing as it goes is what makes _one_ map
enough: a gram runs out exactly when the smaller side's count of it does, so the
larger side's own frequencies are never needed. Cosine gets no such shortcut,
because `Σ b²` needs them.

It also exits the moment `shared` reaches the smaller side's gram count — every
occurrence has been consumed and `Σ min` cannot exceed it. Exact, not a cutoff:
the same number the full walk returns. Worth **2.3–3.7x** where it fires.

Dice routes into it only above `COUNTER_GRAMS` (512) on the larger side and only
at depths 2 and 3 — those are the depths that were measured, and the crossover
moves with the _alphabet_, not just length. The trade it accepts is text with
almost no distinct grams, where sorting thousands of equal keys is nearly free
and a map operation each is not: one repeated gram costs 44–56%, against 90–120%
gained on ordinary text of that length.

### Prepared kernels

A query does not change while a search runs, so its trie is walked once, at
preparation, into flat indexed arrays. Every candidate afterwards is indexed
loops plus one `Map.get` per gram or group — **0.69–0.74x** the cost of walking
both tries per candidate, measured over 100 queries against 1000 prepared bigram
choices. Flattening is specialised to depths 1–3; the trigram kernel measured
0.48x (12-char) and 0.62x (32-char) against the generic walk, which allocates
three stack arrays for every candidate.

**The suffix bound.** `remainingTotals` builds `remaining[i]` = the frequency
still reachable from group `i` onward, so `shared + remaining[i]` is everything
the walk could still find. Under a threshold, a candidate is abandoned the
moment that falls below the minimum. This is the one bound gram counts cannot
express — candidates of the query's own length have an upper bound of 1 however
little they share. Worth **1.11–1.20x** on a search with a limit and a threshold;
`bestMatch` gains 0–7%, because it starts with no cutoff and only raises one as
it goes. The array is `Uint32Array`, not signed: a query past 2³¹ grams would
store a negative bound and reject candidates that qualify.

**Mixed storage is compiled once, never per candidate.** A packed query meeting
its first trie candidate builds `trieFromPacked` and keeps it; a trie query
meeting its first packed candidate builds a `packedProjection` and keeps it, one
per element domain, since a query trie may hold grams of both while a candidate
is packed in exactly one. Deferred rather than eager because most corpora are
all one storage, and a query that never meets the other kind must not pay for
it. The projection keeps only the grams the candidate's domain and radix can
hold — a candidate proves its own elements fit, so a query gram that does not
fit can never match one, and dropping it changes no answer while tightening the
bound.

### The inverted index

Postings are compressed-sparse-row: one `ids` array for the whole index, with
`offsets[ordinal]..offsets[ordinal + 1]` marking each gram's slice. The shape it
replaced carried two typed arrays _per distinct gram_, so seventeen thousand
distinct trigrams meant as many posting objects and twice as many buffers —
object headers proportional to gram variety rather than to the data.

Widths are chosen from the corpus: `Uint16` ids when the corpus fits (≤ 65,536
choices), and the counts array is `null` outright when no gram repeats within any
single choice — which is not the rare case it sounds like, since 99.9% of entries
are `1` on 26-letter trigrams.

**Dense lists.** A posting list covering at least `DENSE_CUTOFF` (2/3) of the
corpus is stored inverted — it holds the choices that _lack_ the gram. `2/3`
rather than the obvious `1/2` because inverting costs a second thing: any query
touching a dense list must score every choice, since a default frequency then
applies to all of them. Writing that out, a dense gram changes the work by
`(N − 2·length + exceptions)` in accumulation and at most `(N − length)` in
selection, and the sum only turns negative above `2N/3`. At exactly one half the
storage saving is zero and the scan is pure loss. Derived rather than swept: real
corpora have discontinuous gram frequencies, and every cutoff from 0.5 to 0.9
performed identically on the ones measured.

**Widening.** The builder starts at the narrowest radix and re-keys the whole
index when a choice does not fit — a loop, not one attempt and a fallback, since
a single choice can need more than one rung (`'\ud800😀'` pushes a byte radix to
BMP and then to full). Each rung is strictly wider than the element that forced
it, so it cannot cycle. Nothing is rolled back on the way round: extraction fills
scratch arrays and only `record` writes to a posting list, so a throw leaves the
index exactly as it was. Re-keying is arithmetic on existing keys, so a late
widening costs the gram _variety_ rather than the corpus.

**Query state is retained and reused**, so nothing per-query is allocated on the
hot path. The accumulator is cleared by walking only what the query touched —
walking the whole thing would put a cost proportional to the corpus back into
every query, which is what this representation exists to avoid — except where a
dense list already made the walk corpus-wide, and then `fill(0)` wins.

There are no membership marks. Where the touched set is read, no dense list was
reached, so every contribution is strictly positive and an untouched entry is
still exactly zero; where a dense list was reached, the set is never read. A
generation-mark array cost 26% of the accumulation loop for a set that is either
unread or already implied.

Ordering matters more than it looks: sorting the touched set measured **84%** of
a `threshold: 0.5` query over 10,000 choices, where accumulation itself was 5%.
So `scan` pays for ascending ids and a ranked call does not — it sorts by score
afterwards and would throw that order away. An unlimited call collects and sorts
once (`O(k log k)`) rather than insertion-placing, which is quadratic with room
for the whole corpus.

### Exactness and refusal

The index reproduces the exhaustive scorer _to the bit_, and refuses rather than
silently disagreeing. Each bound is checked because its failure mode is a wrong
score, not a thrown error:

| guard                        | bound                                 | why                                                                                                                                                 |
| ---------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `assertAddressable`          | 2³²−1 choices, posting entries, grams | the typed arrays would truncate and answer the wrong choice                                                                                         |
| `assertDiceAccumulatorExact` | 2³¹−1 query grams                     | `Σ min` must fit Dice's `Int32Array`                                                                                                                |
| `assertCosineExact`          | `queryGrams × maxChoiceGrams ≤ 2⁵³−1` | a dense list computes `q·(c−1) + q` where a sparse one computes `q·c`; at `q = 116,982,125`, `c = 105,643,526` those differ by 2                    |
| `assertCosineNormsExact`     | both squared norms ≤ 2⁵³−1            | a packed profile sums `Σ c²` per distinct gram where the index sums `2c + 1` per occurrence; one gram repeated 268,435,459 times puts them 16 apart |

The last is a _norm_ bound rather than a length bound deliberately.
`Σ c² ≤ gramCount²` would make `gramCount ≤ 94,906,265` sufficient, and would
refuse a 100-million-gram query of distinct grams whose norm is nowhere near the
boundary. What decides it is repetition, so repetition is what it reads.

---

## Duplication here is deliberate

Several bodies are written twice on purpose, each with the measurement that
decided it recorded beside the code. Do not merge them:

- **`sharedFrequencyTries` / `dotProductTries`**, and **`trieSharedWalk` /
  `trieDotWalk`** — a combiner callback would sit in the innermost frame of the
  walk, where every n-gram metric would make it megamorphic.
- **`DiceIndex` / `CosineIndex`** — separate classes rather than one carrying a
  mode. Dice's overlap is bounded by the query's own gram count, so its
  accumulator is an `Int32Array` where Cosine's must stay `Float64Array`;
  narrowing measured **1.05–1.68x**, the read-modify-write not so much shrinking
  as vanishing. Inlining the score arithmetic into `top` measured a further
  **1.41–1.92x** where closing a callback over locals recovered only 1.04–1.18x.
- **The literal depth-1/2/3 arms throughout.** The generic walk allocates three
  stack arrays per comparison — 1.6x the specialised bigram loop.

Deeper than depth 3 every walk is iterative over an explicit stack, never
recursive: `gramSize` is caller-supplied and equals the trie depth, so recursion
would put a stack overflow inside the range of valid inputs.

---

## Where the rest lives

- **Tests**: colocated, `<module>.test.ts` beside what it covers, with the
  shared oracle in `testing/reference/ngram.ts` and index fixtures in
  `testing/invertedIndex.ts`. The oracle is deliberately the slowest correct
  thing, so a differential failure names the implementation rather than a second
  clever version of it.
- **Benchmarks**: `bench/suites/ngram.bench.ts` and `bench/suites/ngramIndex.bench.ts`. Read
  the `benchmarks` skill before running either.
- **Architecture rules**: `tests/architecture/imports.test.ts`.

**Measuring a change here needs more than `bench:compare`.** The bench suite
runs an esbuild bundle in which this directory is folded flat, while the shipped
library is unbundled one module per source file — so a change that moves a call
across one of these file boundaries cannot show up there at all. Time it against
two `dist/` builds imported into one plain-`node` process instead, alternating
passes, and keep an order control.
