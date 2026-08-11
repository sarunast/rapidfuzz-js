---
title: Damerau-Levenshtein
description: The full four-operation edit distance — insertions, deletions, substitutions, and unrestricted swaps.
---

Damerau-Levenshtein counts four operations — insert, delete, substitute, and
swap adjacent characters — with no restrictions on how they combine. It is
the "mathematically complete" version of what [OSA](/algorithms/osa/)
approximates.

```ts
import { distance, similarity } from 'rapidfuzz-js/damerau-levenshtein'

distance('ca', 'abc') // 2 — swap to 'ac', insert 'b'
similarity('ca', 'abc') // 0.333 (0–1)
```

OSA charges that same pair 3, because its once-edited-stays-edited rule
forbids inserting into an already-swapped pair. That's the entire difference
between the two algorithms — and on realistic typo data, pairs where it
matters are rare.

## Why choose the full version

- **It's a true metric.** Its distances satisfy the triangle inequality
  (`d(a,c) ≤ d(a,b) + d(b,c)`), which some algorithms and data structures —
  metric trees, clustering — depend on. OSA's distances don't.
- **Cross-implementation agreement.** If numbers must match another
  system's "Damerau-Levenshtein", make sure both sides mean the same
  variant — this one is the unrestricted version.

## When to use it

When either bullet above applies. Otherwise [OSA](/algorithms/osa/) gives
the same answers on virtually all real typo data and is the cheaper
algorithm — for everyday "swaps count once" matching, start there.
