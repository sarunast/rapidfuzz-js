---
title: OSA
description: Levenshtein plus adjacent swaps — the practical choice for keyboard typos.
---

Type quickly and you'll produce `teh`, `recieve`, `taht` — adjacent letters
in the wrong order. [Levenshtein](/algorithms/levenshtein/) charges that as
**two** edits (two replacements), which overstates how wrong it is: to a
human it's one slip of the fingers.

OSA — _optimal string alignment_ — adds the missing operation: swapping two
adjacent characters costs **one** edit.

```ts
import { distance } from 'rapidfuzz-js/osa'
import { distance as levenshtein } from 'rapidfuzz-js/levenshtein'

levenshtein('recieve', 'receive') // 2 — two replacements
distance('recieve', 'receive') // 1 — one swap
```

`normalizedSimilarity` reports the same view on 0–1.

## The restriction (and when it bites)

OSA has one rule: **once a piece of the string has been edited, it can't be
touched again.** That keeps the algorithm fast, but on rare inputs it misses
a shorter route:

```ts
import { distance } from 'rapidfuzz-js/osa'

distance('ca', 'abc') // 3 — OSA can't swap and then insert into the pair
```

Full [Damerau-Levenshtein](/algorithms/damerau-levenshtein/) has no such
rule and finds the 2-edit route (swap `ca`→`ac`, insert `b`). For typo-style
data the two agree almost always; they diverge on contrived cases like this
one.

## When to use it

**Human keyboard input** — search boxes, username lookups, form validation —
where transpositions are among the most common errors and each should count
once. Prefer full [Damerau-Levenshtein](/algorithms/damerau-levenshtein/)
only when you need its stricter mathematics: true-metric distances (the
triangle inequality) or number-for-number agreement with another Damerau
implementation.
