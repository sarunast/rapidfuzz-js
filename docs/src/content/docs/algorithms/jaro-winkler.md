---
title: Jaro-Winkler
description: Jaro with a shared-prefix bonus — the standard metric for matching names.
---

People notice the start of a word more than its end — `Jonathan`/`Jonathon`
feel closer than `Jonathan`/`Nathanjo` ever could. Jaro-Winkler encodes that
instinct: it computes [Jaro](/algorithms/jaro/), then **boosts the score for
a shared prefix** of up to four characters.

```ts
import { similarity } from 'rapidfuzz-js/jaro-winkler'
import { similarity as jaro } from 'rapidfuzz-js/jaro'

jaro('martha', 'marhta') // 0.944
similarity('martha', 'marhta') // 0.961 — 'mar' prefix earns the bonus
```

Scores are 0–1; only a `similarity` exists.

## Tuning the bonus

Each shared prefix character adds `prefixWeight × (1 − jaroScore)` to the
score. The default weight is `0.1`; raise it (up to `0.25` — beyond that
scores could exceed 1) to reward prefixes more strongly through a
[scorer](/concepts/scorers/):

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/jaro-winkler'

const heavy = createScorer(similarity, { prefixWeight: 0.2 })
```

## When to use it

The de-facto standard for **matching human names** — record linkage,
deduplication, autocomplete on people or places. Its bias is also its
limitation: strings differing at the very start score noticeably worse, so
for general typo tolerance (where errors land anywhere) prefer
[Levenshtein](/algorithms/levenshtein/) or [OSA](/algorithms/osa/).
