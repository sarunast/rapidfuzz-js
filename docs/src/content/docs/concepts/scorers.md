---
title: Scorers
description: Freeze your comparison decisions — configuration, thresholds, missing values — into one reusable object.
---

A raw metric answers one question about one pair. A real application asks
the same question thousands of times, with the same decisions each time:
how the algorithm is tuned, what counts as "close enough", what to do with
`null`s. A **Scorer** is where those decisions live — made once, applied
everywhere:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const weighted = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

weighted.score('kitten', 'sitting') // 5
```

The scorer also _describes itself_ — useful when code needs to handle any
scorer generically:

```ts
weighted.direction // 'distance' — lower is better
weighted.bounds // [0, Infinity]
weighted.symmetric // true — score(a, b) === score(b, a)
```

Everything that searches — `bestMatch`, `search`, Matchers, the batch
scorers — takes a scorer, not a bare metric. That's the design working as
intended: the search machinery never needs to guess how you want comparisons
made.

## Thresholds: drawing the "good enough" line

Most fuzzy-matching bugs are really threshold bugs — treating a terrible
match as a match because it happened to be the best one available. A
threshold makes the line explicit:

```ts
weighted.score('kitten', 'sitting', { threshold: 3 }) // undefined — 5 edits is too far
```

Three rules, all consequences of the scorer knowing its own direction and
scale:

- For a **similarity**, the threshold is a _minimum_ ("at least 70").
- For a **distance**, it's a _maximum_ ("at most 3 edits").
- It's always in the scorer's own units — `70` for fuzz, `0.7` for a 0–1
  similarity, `3` for an edit distance.

A miss returns `undefined`, not `0` or `-1` — because `0` is a legitimate
score, no sentinel value could be trusted. TypeScript makes you handle the
`undefined`, which is the point.

Two helpers cover the common threshold patterns without a scorer method
call:

```ts
import { isMatch, scoreIfMatch } from 'rapidfuzz-js'

scoreIfMatch(weighted, 'kitten', 'sitting', { threshold: 5 }) // 5
isMatch(weighted, 'kitten', 'sitting', { threshold: 3 }) // false
```

:::tip
Thresholds aren't only a filter — they're a speed lever. The edit-distance
algorithms use the threshold as a cutoff and abandon a pair as soon as it
can no longer qualify. A meaningful threshold makes searches _faster_, not
just cleaner.
:::

## Missing values

Real data has `null`s. Only `null` and `undefined` count as missing — and
similarity scorers treat a missing operand as "matches nothing":

```ts
import { createScorer } from 'rapidfuzz-js'
import { weightedRatio } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(weightedRatio)
scorer.score(null, 'text') // 0
```

Two missing operands score `0` as well, not `100`. Unknown compared with
unknown isn't evidence of a match, and scoring it perfect would sort every
missing record to the top of a search:

```ts
scorer.score(null, null) // 0
```

That default keeps a search over gappy data running, with missing entries
naturally scoring at the bottom. If a `null` reaching the scorer means a bug
upstream, opt into strictness:

```ts
const strict = createScorer(weightedRatio, { missing: 'throw' })
strict.score(null, 'text') // throws TypeError
```

Distance scorers always throw on missing operands — there's no honest
"worst possible distance" to report when distances are unbounded.

One boundary worth knowing: _missing_ is lenient by default, _invalid_ never
is. Empty strings are valid (they're just empty). Numbers, booleans, and
objects without an array-like `length` always throw — they're type errors in
your data, not gaps in it.

## Options objects reject unknown keys

A misspelled option is a `TypeError`, not a silently ignored property:

```ts
search('new york', teams, { scorer, limitt: 2 })
// TypeError: unknown search option 'limitt'
```

This covers the searches, the Matcher methods, `scoreMatrix`, `scorePairs`,
and `prepareChoice`. The failure it prevents is the quiet one — a typo'd
`threshold` or `limit` that leaves you looking at unfiltered results and
wondering why the option "didn't work".

## One more thing a scorer can do

A scorer can convert a candidate into its internal form ahead of time —
`scorer.prepareChoice` returns an opaque handle you store beside your own
data, so repeated searches stop re-preparing every candidate:
[Prepared choices](/guides/prepared-choices/).
