---
title: Cosine
description: The angle between two n-gram frequency vectors — Dice's sibling, with repeats weighted heavily.
---

Cosine reads the same n-gram bag as [Sørensen-Dice](/algorithms/dice/) — both
chop the inputs into overlapping runs of `gramSize` elements, two by default —
and combines it differently. Each side becomes a **frequency vector** over the
grams it contains, and the score is the cosine of the angle between them:

```ts
import { similarity } from 'rapidfuzz-js/cosine'

similarity('night', 'nacht') // 0.25 — one shared gram, `ht`
similarity('banana', 'bananas') // 0.949
similarity('new york mets', 'mets new york') // 0.833
```

```text
                     Σ a_g · b_g
Cosine(A, B) = ─────────────────────────
                     ‖A‖ · ‖B‖
```

Scores are 0–1, with 1 identical, and `distance` is `1 - similarity`. Like
Dice, Cosine is normalized by construction, so `normalizedSimilarity` and
`normalizedDistance` are the same functions under the names the other
algorithms use.

## Not the Cosine some libraries ship

Two things about this definition surprise people arriving from elsewhere.

**It is the dot product of the frequency vectors**, not the
intersection-count formula `|A ∩ B| / sqrt(|A| · |B|)` — Otsuka-Ochiai — that
several JavaScript and Python packages publish under the name "cosine". The
two always agree when no gram repeats, and can part company once one does —
though not always, since a profile scored against itself is `1` either way.

**Repeats therefore dominate.** A gram occurring three times against twice
contributes six, where Dice counts two. That is the whole difference between
the two metrics on repetitive input:

```ts
import { similarity as cosine } from 'rapidfuzz-js/cosine'
import { similarity as dice } from 'rapidfuzz-js/dice'

cosine('ababab', 'abab') // 0.992
dice('ababab', 'abab') // 0.75
```

Every other choice is shared with Dice, and the
[Dice page](/algorithms/dice/#three-choices-other-implementations-make-differently)
spells them out: multiset rather than set, no padding at the ends, and a
defined answer for inputs shorter than one gram — equal ones score `1`,
anything else `0`.

## Choosing the gram size

`gramSize` is scorer configuration, exactly as for Dice:

```ts
import { createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/cosine'

createScorer(similarity, { gramSize: 3 }).score('banana', 'bananas') // 0.926
```

A `gramSize` below `1`, or one that is not a _safe_ integer, is a `RangeError`
— `1e300` is an integer, and a trie that deep is not a request anyone means.
Elements are compared by identity, so arrays of anything work and astral
characters compare as whole code points.

## Searching, and what it costs

Cosine has **no upper bound computable from gram counts**, which is the one
practical asymmetry with Dice: every candidate's profile has to be built
before a threshold can reject it. Over a long candidate list under a high
cutoff, Dice is the cheaper of the two by a wide margin: 100 length-skewed
pairs at `threshold: 0.8` measure 0.005 ms through Dice's bound against
4.2 ms through Cosine, which builds both profiles every time.

Preparing choices removes most of that from repeated searches; see
[Prepared choices](/guides/prepared-choices/). A prepared choice carries its
metric and gram size, so a Cosine scorer refuses one built by a Dice scorer,
and a `gramSize: 3` scorer refuses one built at the default.

## Searching a large collection

Dice, Cosine and Tversky can all be searched through
[`createIndexedMatcher`](/concepts/matchers/#indexed-matchers),
which builds one inverted n-gram index over the collection instead of preparing
each choice. It is the same Matcher afterwards, with the same exact results, and
on 10,000 file paths it measured 11–13x faster per query and 5x smaller —
except on a query made of grams that nearly every choice shares, which is where
the idea stops paying and measured 0.7x.

## When to use it

Reach for Cosine over Dice when **how often** a gram appears is signal rather
than noise: longer text, repetitive identifiers, sequences where a doubled
run means something. It is also the more familiar formula if your thresholds
were tuned against a vector-space text-similarity tool.

Reach for [Dice](/algorithms/dice/) otherwise. It answers the same question
about overlap, is less swayed by one repeated gram, and prunes on length
before doing any work — which matters as soon as you are scoring a query
against thousands of candidates.
