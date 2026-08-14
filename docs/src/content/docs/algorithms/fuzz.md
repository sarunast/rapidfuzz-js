---
title: Fuzz
description: The 0–100 family for messy real-world text — partial matches, word order, and the do-the-right-thing metric.
---

Edit distances compare _characters_. But most real-world mismatches aren't
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

**Whole strings are comparable** → `ratio`. The baseline: normalized
Indel scaled to 0–100.

```ts
import { ratio } from 'rapidfuzz-js/fuzz'
ratio('this is a test', 'this is a test!') // 96.55
```

**One string may contain the other** → `partialRatio`. Slides the
shorter string over the longer and reports the best window — so a substring
scores ~100 even when the surrounding text differs wildly.

**Same words, different order** → `tokenSortRatio`. Splits into words,
sorts them, compares — word order stops mattering entirely.

```ts
import { tokenSortRatio } from 'rapidfuzz-js/fuzz'
tokenSortRatio('fuzzy wuzzy was a bear', 'wuzzy fuzzy was a bear') // 100
```

**Overlapping words, different amounts of extra** → `tokenSetRatio`.
Compares the word _sets_, factoring out the words both sides share — one
side having extra words hurts far less — far enough less that containment
scores a flat `100`:

```ts
tokenSetRatio('data engineer', 'data engineer cloud platform') // 100
tokenSortRatio('data engineer', 'data engineer cloud platform') // 63.41
```

That's the right opinion for a company name and the wrong one for a job
title. [Matching records](/guides/matching-records/) works through the
difference.

**Combinations** — `tokenRatio` takes the better of sort/set;
`partialTokenSortRatio`, `partialTokenSetRatio`, and
`partialTokenRatio` apply the token strategies over the best partial
window.

**Just handle it** → `weightedRatio`. Tries the appropriate strategies
per pair, weights them by how the lengths compare, and reports the best —
the port of RapidFuzz's famous `WRatio`. **This is the right default**:
start here, inspect real mismatches, and only pin a specific metric when
you can name what `weightedRatio` gets wrong.

Coming from Python RapidFuzz? The names are the same ones, in camelCase:
`ratio`, `partial_ratio` → `partialRatio`, `token_sort_ratio` →
`tokenSortRatio`. The one that is not a transliteration is `WRatio`, which
here is `weightedRatio`; `QRatio` has no counterpart at all.

Coming from rapidfuzz-js 0.11? These scorers were called `similarity`,
`tokenSortSimilarity`, `weightedSimilarity` and so on until 0.12.0, which gave
the whole subpath RapidFuzz's vocabulary. Replace `…Similarity` with `…Ratio`,
and plain `similarity` with `ratio`; the metric brands changed to match, and no
score moved. The [README](https://github.com/sarunast/rapidfuzz-js#upgrading-from-011)
has the full table.

## Empty inputs: the family disagrees on purpose

Compare two empty strings and the answer depends on which fuzz metric you
asked:

```ts
ratio('', '') // 100
tokenSortRatio('', '') // 100
tokenSetRatio('', '') // 0
partialTokenSetRatio('', '') // 0
weightedRatio('', '') // 0
```

The set-based scorers intersect _sets of tokens_, and a side with no tokens
has no set to intersect — so they report `0` where the character-level and
sort-based scorers report `100`. This isn't a bug being preserved by
accident: FuzzyWuzzy answered `0` here, RapidFuzz kept it deliberately
(upstream issue 110), and this port matches so that scores stay comparable
across all three.

Whitespace-only input is where `weightedRatio` splits from the two set
scorers — `weightedRatio('   ', '   ')` is `100`, while
`tokenSetRatio('   ', '   ')` stays `0`.

If empty values are meaningful in your data, filter them before scoring
rather than reasoning about which scorer does what.

## Two things fuzz metrics don't do

- **Preprocess.** Unlike Python RapidFuzz's optional `processor`, nothing
  is lowercased or stripped for you. `'Apple'` vs `'apple!'` scores as two
  real differences unless you normalize — pair fuzz metrics with
  [`normalizeText`](/guides/preprocessing/) to get upstream's
  `default_process` behaviour.
- **Explain.** They return a score, not the why. The one exception:
  `partialRatioAlignment` reveals where the best partial window sat —
  `{ score, srcStart, srcEnd, destStart, destEnd }` — for highlighting what
  matched.

## When to use the family

Any time you're matching **text meant for humans** — product titles, names
with honorifics, addresses, song titles, free-form user input. For codes,
identifiers, and single words, the character-level algorithms
([Levenshtein](/algorithms/levenshtein/), [OSA](/algorithms/osa/)) are the
sharper tool.
