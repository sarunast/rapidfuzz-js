---
title: Fuzz
description: The 0–100 family for messy real-world text — partial matches, word order, and the do-the-right-thing metric.
---

Edit distances compare *characters*. But most real-world mismatches aren't
character noise — they're **structural**: the same words in a different
order, one string containing the other, extra words on one side.

```text
"Smith, John"              vs  "John Smith"
"NY Jets"                  vs  "New York Jets (NFL)"
"wireless keyboard black"  vs  "black wireless keyboard"
```

Levenshtein scores these poorly — many characters are "wrong" even though
every word is right. The fuzz family exists for exactly this. All nine
metrics score 0–100 (higher is better), built on the
[Indel](/algorithms/indel/) similarity, and each handles one kind of
structural mismatch.

## The metrics, by the problem they solve

**Whole strings are comparable** → `similarity`. The baseline: normalized
Indel scaled to 0–100.

```ts
import { similarity } from 'rapidfuzz-js/fuzz'
similarity('this is a test', 'this is a test!') // 96.55
```

**One string may contain the other** → `partialSimilarity`. Slides the
shorter string over the longer and reports the best window — so a substring
scores ~100 even when the surrounding text differs wildly.

**Same words, different order** → `tokenSortSimilarity`. Splits into words,
sorts them, compares — word order stops mattering entirely.

```ts
import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'
tokenSortSimilarity('fuzzy wuzzy was a bear', 'wuzzy fuzzy was a bear') // 100
```

**Overlapping words, different amounts of extra** → `tokenSetSimilarity`.
Compares the word *sets*, factoring out the words both sides share — one
side having extra words hurts far less.

**Combinations** — `tokenSimilarity` takes the better of sort/set;
`partialTokenSortSimilarity`, `partialTokenSetSimilarity`, and
`partialTokenSimilarity` apply the token strategies over the best partial
window.

**Just handle it** → `fuzzySimilarity`. Tries the appropriate strategies
per pair, weights them by how the lengths compare, and reports the best —
the port of RapidFuzz's famous `WRatio`. **This is the right default**:
start here, inspect real mismatches, and only pin a specific metric when
you can name what `fuzzySimilarity` gets wrong.

Coming from Python RapidFuzz? The mapping is mechanical: `ratio` →
`similarity`, `partial_ratio` → `partialSimilarity`, `token_sort_ratio` →
`tokenSortSimilarity`, `WRatio` → `fuzzySimilarity`, and so on.

## Two things fuzz metrics don't do

- **Preprocess.** Unlike Python RapidFuzz's optional `processor`, nothing
  is lowercased or stripped for you. `'Apple'` vs `'apple!'` scores as two
  real differences unless you normalize — pair fuzz metrics with
  [`normalizeText`](/guides/preprocessing/) to get upstream's
  `default_process` behaviour.
- **Explain.** They return a score, not the why. The one exception:
  `partialSimilarityAlignment` reveals where the best partial window sat —
  `{ score, srcStart, srcEnd, destStart, destEnd }` — for highlighting what
  matched.

## When to use the family

Any time you're matching **text meant for humans** — product titles, names
with honorifics, addresses, song titles, free-form user input. For codes,
identifiers, and single words, the character-level algorithms
([Levenshtein](/algorithms/levenshtein/), [OSA](/algorithms/osa/)) are the
sharper tool.
