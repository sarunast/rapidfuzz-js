---
title: Levenshtein
description: "The classic edit distance: how many single-character fixes turn one string into the other?"
---

The Levenshtein distance answers the most intuitive similarity question
there is: **how many single-character fixes would it take to turn one string
into the other?** A fix is inserting a character, deleting one, or replacing
one.

```text
kitten → sitten   (replace k with s)
sitten → sittin   (replace e with i)
sittin → sitting  (insert g)
```

Three fixes — so the distance is 3. No shorter route exists, and the
algorithm guarantees it finds the minimum.

```ts
import { distance, similarity } from 'rapidfuzz-js/levenshtein'

distance('kitten', 'sitting') // 3
similarity('kitten', 'sitting') // 0.571 — the same fact as a 0–1 score
```

Use `distance` when the count itself is meaningful ("reject if more than 2
edits"); use `similarity` when comparing across strings of different lengths
— 3 edits is bad for a 6-letter word and trivial for a paragraph, and the
0–1 score accounts for that.

## Tuning the costs

By default every operation costs 1. If your domain considers some errors
worse than others, weight them through a [scorer](/concepts/scorers/):

```ts
import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const weighted = createScorer(distance, {
  weights: { insertion: 1, deletion: 1, substitution: 2 },
})

weighted.score('kitten', 'sitting') // 5 — two substitutions now cost 4
```

Weights also accept a tuple: `[insert, delete, replace]`. The
substitution-costs-2 setting is common enough to have its own name — it's
exactly [Indel](/algorithms/indel/) distance.

## Seeing the edits

`editops` and `opcodes` list the actual operations behind the number — for
highlighting differences or building diff views:

```ts
import { editops } from 'rapidfuzz-js/levenshtein'

editops('kitten', 'sitting')
// { tag: 'replace', srcPos: 0, destPos: 0 }, ... 
```

See [Comparing strings](/guides/comparing-strings/#recovering-the-edits).

## When to use it

The default for **typo tolerance on whole strings**: misspelled words, OCR
output, close variants of codes and identifiers. Reach elsewhere when:

- Swapped adjacent letters (`teh`) should count as *one* mistake →
  [OSA](/algorithms/osa/).
- Word order or extra words are the real difference →
  [Fuzz](/algorithms/fuzz/) token metrics.
- You're matching person names → [Jaro-Winkler](/algorithms/jaro-winkler/).
