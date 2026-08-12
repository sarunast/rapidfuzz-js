---
title: LCS
description: Longest common subsequence — similarity as "what do these share, in order?"
---

The longest common subsequence is the longest run of characters appearing in
**both** strings **in the same order** — not necessarily side by side.
`ABCBDAB` and `BDCABA` share `BDAB` (among others of length 4): pick out
those letters in either string and they appear in that order.

Instead of asking "how many edits apart?", LCS asks **"how much do they
share?"** — the glass-half-full view of the same comparison.

```ts
import { distance, normalizedSimilarity, similarity } from 'rapidfuzz-js/lcs'

distance('ABCBDAB', 'BDCABA') // 3 — characters outside the shared subsequence
similarity('ABCBDAB', 'BDCABA') // 4 — the length of the common subsequence itself
normalizedSimilarity('ABCBDAB', 'BDCABA') // 0.571 — that length over the longer input
```

LCS is the one algorithm where the raw `similarity` is the headline number:
it _is_ the length of the longest common subsequence.

## Seeing the alignment

```ts
import { editops, opcodes } from 'rapidfuzz-js/lcs'

editops('ABCBDAB', 'BDCABA')
// 'insert' and 'delete' operations around the common subsequence
```

`opcodes` includes the `equal` stretches — the shared runs themselves —
which is the raw material of a diff view.

## When to use it

When shared content matters more than the differences between it: comparing
texts that may contain long unrelated insertions, lining up sequences for
diff-style output, or any place "what survived between version A and B?" is
the real question. [Indel](/algorithms/indel/) is the edit-distance view of
exactly the same quantity — same information, opposite framing.
