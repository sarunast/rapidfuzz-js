---
title: Prefix and Postfix
description: The simplest metrics — how do the strings start, or end?
---

Two deliberately simple metrics: `rapidfuzz-js/prefix` measures the run of
identical characters at the **start** of both strings, `rapidfuzz-js/postfix`
the run at the **end**. No alignment, no edits — just "how far do they agree
from this side?"

```ts
import { distance, normalizedSimilarity, similarity } from 'rapidfuzz-js/prefix'

similarity('apple', 'applesauce') // 5 — the shared run is 5 characters long
normalizedSimilarity('apple', 'applesauce') // 0.5 — 5 shared of the longer 10
distance('apple', 'applesauce') // 5 — the characters outside the shared run
```

```ts
import { normalizedSimilarity as postfix } from 'rapidfuzz-js/postfix'

postfix('walking', 'running') // 0.429 — shared 'ing' against length 7
```

`similarity` is the length of the shared run; `normalizedSimilarity` divides
it by the longer input (0–1); `distance` counts everything outside the run.

## When to use it

- **As a cheap pre-filter.** Prefix comparison is drastically cheaper than
  any edit distance. When candidates that don't even start alike can be
  ruled out — autocomplete, sorted-list narrowing — filter on prefix first
  and spend the real algorithm on survivors.
- **When the domain really is the edge.** Version strings, file paths,
  hierarchical codes, or suffixes like file extensions — places where "how
  far do they agree from the start/end" _is_ the question, not a proxy
  for it.

For general similarity these are the wrong tool — `prefix` scores
`'apple'` vs `'napple'` at 0, one insertion notwithstanding. That blindness
is what makes them fast; use them where it can't mislead.
