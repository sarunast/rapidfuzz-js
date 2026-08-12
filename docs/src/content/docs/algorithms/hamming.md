---
title: Hamming
description: Position-by-position comparison for fixed-format data — no alignment, ever.
---

Hamming is the simplest comparison here: walk both sequences in lockstep and
**count the positions where they differ**. Nothing shifts, nothing aligns —
position 3 compares with position 3, full stop.

```ts
import { distance, normalizedSimilarity } from 'rapidfuzz-js/hamming'

distance('karolin', 'kathrin') // 3 — positions 2, 3, 4 differ
normalizedSimilarity('karolin', 'kathrin') // 0.571 (0–1)
```

That rigidity is the point — and the trap. Insert one character near the
front of a string and every later position shifts: Hamming sees almost
nothing in common, while [Levenshtein](/algorithms/levenshtein/) correctly
sees one edit. Hamming is only meaningful when **positions themselves carry
meaning** and inputs can't shift.

## Unequal lengths

Position-wise comparison needs equal lengths, so by default the shorter
input is treated as padded and the trailing overhang counts as differences:

```ts
import { distance } from 'rapidfuzz-js/hamming'

distance('abc', 'abcd') // 1 — the unmatched 'd'
```

If unequal lengths mean your data is wrong rather than merely different,
turn the padding off through a [scorer](/concepts/scorers/) and let it
throw:

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/hamming'

const strict = createScorer(distance, { pad: false })
strict.score('abc', 'abcd') // throws — sequences are not the same length
```

## Seeing the differences

`editops` and `opcodes` report the differing positions — substitutions,
plus insertions for a padded overhang.

## When to use it

Fixed-format, aligned data: hashes and checksums, fixed-width codes,
barcodes, equal-length DNA reads, error detection in transmissions. If your
data can gain or lose characters — which includes essentially all human
input — use [Levenshtein](/algorithms/levenshtein/) instead.
