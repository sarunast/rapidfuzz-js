---
title: Jaro
description: Similarity built for short strings and names — generous with partial agreement.
---

Edit distance treats a 6-letter name harshly: two edits in `martha` is a
third of the string gone. The Jaro similarity was designed for exactly this
case — **short strings, especially names**, where partial agreement should
still score well. It comes from the world of record linkage: matching people
across census rolls and databases.

Instead of counting edits, Jaro looks for **matching characters** — the same
character on both sides, close enough in position — and then checks how many
of those matches are **in the wrong order** (transpositions):

```ts
import { similarity } from 'rapidfuzz-js/jaro'

similarity('martha', 'marhta') // 0.944 — all letters match, one pair swapped
similarity('dixon', 'dicksonx') // 0.767
```

Scores are 0–1, with 1 identical. "Close enough in position" scales with
length: characters match if they sit within half the longer string's length
of each other. There is only a `similarity` — no distance form.

## When to use it

Person and place names, short identifiers, deduplicating records — anywhere
strings are short and a few differences shouldn't crater the score. In
practice most users want [Jaro-Winkler](/algorithms/jaro-winkler/), which
adds one refinement that fits names even better: differences at the start
hurt more than differences at the end.
