---
title: Hamming
description: Position-by-position comparison for fixed-format data — no alignment, ever.
---

Hamming is the simplest comparison here: walk both sequences in lockstep and
**count the positions where they differ**. Nothing shifts, nothing aligns —
position 3 compares with position 3, full stop.

```ts
import { distance, similarity } from 'rapidfuzz-js/hamming'

distance('karolin', 'kathrin') // 3 — positions 2, 3, 4 differ
similarity('karolin', 'kathrin') // 0.571 (0–1)
```

That rigidity is the point — and the trap. Insert one character near the
front of a string and every later position shifts: Hamming sees almost
nothing in common, while [Levenshtein](/algorithms/levenshtein/) correctly
sees one edit. Hamming is only meaningful when **positions themselves carry
meaning** and inputs can't shift.

## Unequal lengths

Since position-wise comparison needs equal lengths, unequal inputs throw by
default. If trailing overhang should simply count as differences, opt into
padding through a [scorer](/concepts/scorers/):

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/hamming'

const padded = createScorer(distance, { pad: true })
padded.score('abc', 'abcd') // 1
```

## Seeing the differences

`editops` and `opcodes` report the differing positions — substitutions,
plus insertions for a padded overhang.

## When to use it

Fixed-format, aligned data: hashes and checksums, fixed-width codes,
barcodes, equal-length DNA reads, error detection in transmissions. If your
data can gain or lose characters — which includes essentially all human
input — use [Levenshtein](/algorithms/levenshtein/) instead.
