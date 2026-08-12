---
title: Indel
description: Edit distance with only insertions and deletions — the measure underneath the fuzz family.
---

Indel is [Levenshtein](/algorithms/levenshtein/) with one operation removed:
you may **insert** and **delete**, but not replace. Fixing a wrong character
takes two steps — delete the wrong one, insert the right one.

```ts
import { distance, normalizedSimilarity } from 'rapidfuzz-js/indel'

distance('lewenstein', 'levenshtein') // 3
normalizedSimilarity('abc', 'axc') // 0.667 — the b↔x swap costs 2 of 6
```

Why remove an operation? Because what's left measures something specific:
**how much of the two strings is shared**. Indel is the mirror image of the
[longest common subsequence](/algorithms/lcs/) — everything that isn't part
of the common subsequence must be deleted or inserted:

```text
distance(a, b) = len(a) + len(b) − 2 × lcs(a, b)
```

That "shared content" view of similarity turns out to be the right basis for
comparing _text_ (as opposed to codes or identifiers), which is why the
whole [fuzz family](/algorithms/fuzz/) — `similarity`, the token metrics,
`weightedSimilarity` — is normalized Indel scaled to 0–100.

## Seeing the edits

```ts
import { editops } from 'rapidfuzz-js/indel'

editops('kitten', 'sitting')
// only 'insert' and 'delete' tags — never 'replace'
```

## When to use it

Directly, when you want diff-like similarity — "how much is shared, in
order" — or need edit operations that never merge a delete+insert into a
replace. Indirectly, you use it every time you call a fuzz metric; reach for
raw Indel when you want that same opinion without the 0–100 scaling.
