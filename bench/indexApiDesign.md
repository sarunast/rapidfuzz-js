# `createIndexedMatcher` — the Stage C design

What Stage B proved is in `indexProgress.md`. This is how it becomes `src/`.

The shape in one line:

```ts
const matcher = createIndexedMatcher(files, {
  scorer: createScorer(diceSimilarity, { gramSize: 3 }),
  getText: (file) => file.path,
})
matcher.search('src/algorthms/dice.ts', { limit: 5, threshold: 0.5 })
```

**A separate constructor, and the same `Matcher` afterwards.** Explicit at the
one point where the decision has consequences — construction strategy, what is
retained, which scorers are accepted, what a rebuild costs — and invisible at
every point after it. `best`, `search`, `searchIter`, `size` and `scorer` are the
same members returning the same `Match` objects, so swapping the constructor is
the whole diff at a call site. Making `createMatcher` choose silently is Stage D
and needs a predictor; making the query API different would be a second API to
learn for the same question.

## The dependency problem, and the capability that solves it

`tests/architecture/imports.test.ts:40` forbids `search/` from importing
`algorithms/`. An inverted n-gram index is algorithm knowledge. Both are correct
and they have to meet somewhere.

They meet on `MetricCompilation`, which already exists in `src/core/protocol.ts`
and already carries exactly this kind of thing — `prepareQuery`, `prepareChoice`,
`preparedChoiceKey`. One optional member joins them:

```ts
// core/protocol.ts
interface Compilation<TDirection extends Direction, TBrand = AnyBrand> {
  // …existing members…

  /**
   * Builds one corpus-wide representation for the whole collection, replacing
   * the per-choice `prepareChoice` handles a Matcher would otherwise retain.
   * Absent on every metric that has no such representation, which is what
   * `createIndexedMatcher` refuses on.
   */
  readonly indexChoices?: ((choiceCount: number) => ChoiceIndexBuilder) | undefined
}

export interface ChoiceIndexBuilder {
  /** Ids arrive in ascending order from `0`, one per accepted choice. */
  add(id: number, choice: Sequence): void
  seal(): ChoiceIndex
}

export interface ChoiceIndex {
  /**
   * Every id that qualifies, ordered by `(score desc, id asc)` — the order
   * `topSimilarity` produces, since a stored id is its `order`.
   *
   * Borrows: the arrays are the index's own scratch and are valid until the
   * next call, like `prepareChoice`'s borrowed sequence. `length` is the count,
   * the arrays may be longer.
   */
  select(query: Sequence, threshold: number | null, limit: number | null): SelectedChoices
}

export interface SelectedChoices {
  readonly ids: Uint32Array
  readonly scores: Float64Array
  readonly length: number
}
```

`search/` sees a factory of numbers. It never learns what a gram is.

The contract's real content is three obligations the parity tests exist to hold
the implementation to, and they are the reason this is a capability rather than
a hint:

1. **Exact.** `select` returns what scoring every choice would have returned —
   the same scores, to the bit.
2. **Ordered.** `(score desc, id asc)`, which is `topSimilarity`'s rule with
   `order` = stored id.
3. **Complete.** At `threshold: null` or `0`, choices that share nothing with the
   query still score `0` and still come back. The index has to zero-fill; it may
   not answer only what it touched.

Obligation 3 is the one that looks skippable and is not — the exhaustive drivers
return those rows today, and a caller diffing the two implementations would find
it immediately.

## Files

| File                                             | Change                                                                  |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| `src/core/protocol.ts`                           | `indexChoices?`, `ChoiceIndexBuilder`, `ChoiceIndex`, `SelectedChoices` |
| `src/search/indexedMatcher.ts`                   | new — `createIndexedMatcher`                                            |
| `src/search/types.ts`                            | `IndexedMatcherOptions`                                                 |
| `src/search/index.ts`, `src/index.ts`            | export both                                                             |
| `src/algorithms/shared/ngram/inverted/`          | new — the CSR posting store, from `bench/tooling/ngramIndex.ts`         |
| `src/algorithms/shared/metricAdapter.ts`         | pass an optional `indexChoices` through `builtInMetric`                 |
| `src/algorithms/{dice,cosine}/implementation.ts` | declare theirs                                                          |

The index lands in `algorithms/shared/` because that is where "proven low-level
data structures" belong and it is now proven; Dice and Cosine both import it, so
the reachability guard at `imports.test.ts:89` is satisfied through
`algorithms/dice/index.ts`. `search/` gains no import at all.

## Which scorers qualify, and how a caller finds out

**Direction is static. Indexability is not.**

`IndexedMatcherOptions` constrains `TDirection extends 'similarity'`, so a
distance scorer is a compile error rather than a throw — Stage B measured only
the similarity direction, and top-k under distance is a different driver.

Indexability itself is a **runtime** `TypeError` naming the scorer:

```
createIndexedMatcher: this scorer has no indexed representation.
Indexed search is available for dice.similarity and cosine.similarity.
```

Three reasons not to make it static, in the order they bind:

- A custom scorer built from `(a, b) => number` can never be enumerated in a
  type, and refusing to type-check one is worse than throwing on it.
- The brand is the metric id literal, so the type-level spelling would be
  `TBrand extends 'dice.similarity' | 'cosine.similarity'` **written inside
  `search/`** — algorithm identity migrating into shared infrastructure, which
  is the inversion `imports.test.ts` protects at runtime and cannot see at the
  type level. Adding a third algorithm would mean editing `search/`.
- Qualification is a property of the compiled scorer, not the metric: it is
  produced inside `compile(configuration)`, so a future configuration that
  disqualifies itself has somewhere to say so.

The cost is honest and small — one throw, at construction, on the line that
opted in.

## What is retained

This is where the design earns its measurements. On the 12,947-path corpus a
Matcher retains 251.0 MB, of which the profiles are 250.2 MB and the per-choice
rows are 0.77 MB — **0.3%, and beneath notice.** Beside a 4.82 MB index the same
0.77 MB is **14%**, and it is the largest thing left. Removing the representation
promotes the bookkeeping.

So an indexed matcher does not keep `StoredItem[]`:

```ts
const items: TItem[] // dense, position === id
const keys: TKey[] | null // null when key === id
const index: ChoiceIndex
```

`keys` is `null` for an array collection with no skipped gaps — the common case —
which is the difference between a second parallel array and nothing. Flat arrays
instead of one object per choice takes the bookkeeping from ~56 B to ~8 B per
choice, ~725 KB to ~104 KB on that corpus: **14% of the total becomes 2%.**

Nothing else is retained. In particular no `PreparedChoice`, which is the claim
the whole exercise rests on, and it is worth a test that walks the built matcher
and fails on finding one — the bench harness has that check already
(`retainsProfile`) and it caught nothing only because nothing was wrong.

## Semantics: what is identical, and the two things that are not

Identical, and tested against `createMatcher` row for row: scores, ordering, tie
breaks, `key` for every collection shape, `normalize` on both sides,
`missingItems`, the missing-query path, `threshold` null/0/positive, `limit`
null/0/k, and gramless choices and queries.

Two differences, both worth stating in the JSDoc rather than discovering:

- **`searchIter` stops being lazy.** Today it scores in stored order and yields
  as it goes, so a caller who breaks early pays only for what they consumed. An
  index has no meaningful partial state — accumulation is the work — so the
  indexed version accumulates, then yields. Same values, same order, no early
  exit. It stays on the interface because removing a member would break the
  "swap the constructor" promise for the sake of a saving only some callers use.
- **`limit: null` with no threshold is `O(N)` output.** It already is on the
  exhaustive path — the result _is_ N matches — but on an index it is also the
  one query shape where zero-filling dominates the query itself.

## Refusals

Each throws a `TypeError` at construction, with the reason in the message:

- **`getPrepared`.** A prepared handle is the representation being replaced;
  ingesting handles would mean retaining them. Declared `getPrepared?: undefined`
  on `IndexedMatcherOptions`, the way `MatcherOptions` already declares it, so
  most callers see it as a type error first.
- **A non-integer element.** Gram elements must be integers — code points, in
  practice. Strings always qualify. An array of objects, or one containing `NaN`,
  does not, and the exhaustive matcher handles both. The message names the
  offending choice's key. Element interning would lift this and is not Stage C.
- **A collection larger than the representation's 32-bit bounds.** `ids`,
  `offsets` and `gramCount` are `Uint32Array`; nothing realistic approaches
  4.29e9 choices or posting entries, and a library should throw rather than
  truncate into a typed array.

## Order of work

Each step is committable and leaves the suite green.

1. **`ngramIndex.ts` into `algorithms/shared/`**, with the prototype's tests
   ported as `tests/algorithms/ngramIndex.test.ts` — parity against
   `matcher.search`/`.best` on key, score and order is the whole test, and Stage
   B's fixed corpora and configuration matrix port directly. 100% coverage with
   no `v8 ignore` is the gate; the prototype has bench-only branches (statistics,
   counters, the string key scheme) that either earn a test or are deleted on the
   way in.
2. **The protocol members**, unimplemented by anyone. Type-only, so nothing moves.
3. **Dice and Cosine declare `indexChoices`** through `builtInMetric`. Still
   nothing consumes it.
4. **`createIndexedMatcher`**, and the differential test that runs a corpus
   through both constructors and asserts identical results.
5. **Exports and docs**, `README` and the site, with the adverse case stated
   beside the win.
6. **`bench/ngramIndex.bench.ts` repointed** at the real thing, and
   `bench/tooling/ngramIndex.ts` deleted — the prototype's job ends when the
   thing it was predicting exists.

## Deferred, on purpose

- **`createMatcher(items, { index: 'auto' })`.** Needs the `weightedShare`
  predictor and a decision cheap enough to make before building either
  representation. The explicit constructor has to prove itself first.
- **Sharing one index across scorers.** The representation carries both
  `gramCount` and `squaredNorm`, so it already serves Dice and Cosine, and
  `squaredNorm` costs Dice nothing. Exposing that as a reusable handle means a
  new public representation type, which the API rules refuse until there is a
  measured reason.
- **Incremental update.** Rebuild-only, as `createMatcher` already is.
- **The two-pass CSR build**, until peak build memory is a limit. It is measured
  and it is not.
- **Cosine prefix filtering** (WAND/MaxScore), and every posting-layout
  optimisation in `indexProgress.md`'s open list. Full accumulation is the
  architectural win; those are increments on top of a thing that exists.
