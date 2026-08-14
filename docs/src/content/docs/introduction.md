---
title: Introduction
description: What fuzzy matching is, why exact comparison fails, and how rapidfuzz-js solves it.
---

## The problem

Computers compare text literally. To `===`, these are simply different
strings:

```text
"John Smith"      vs  "john smith"
"receive"         vs  "recieve"
"New York Jets"   vs  "new york jet"
"Main St."        vs  "Main Street"
```

A human sees four obvious matches. Search boxes, "did you mean?" suggestions,
importing messy spreadsheets, deduplicating customer records — all of these
need a computer to see what the human sees: _these strings are almost the
same_.

**Fuzzy matching** replaces the yes/no of `===` with a score: instead of
"equal or not", you get "94% similar" or "2 edits apart", and you decide how
close is close enough.

## What rapidfuzz-js is

rapidfuzz-js is a fuzzy matching library for JavaScript and TypeScript. It is
a port of [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), one of the
most widely used fuzzy matching libraries in the Python world — the same
algorithms, producing the same numeric results for the same inputs, so
decades of collective experience with these algorithms carries over directly.

It runs in Node.js 22+, browsers, and edge runtimes, has no dependencies, and
ships strict TypeScript types. Each algorithm lives on its own import path,
so your bundle only contains what you actually use.

## Three building blocks

The whole library is three ideas, stacked:

```text
Metric → Scorer → Matcher
```

**A [Metric](/concepts/metrics/) measures one pair.** It's a plain function:
give it two strings, get a number back.

```ts
import { distance } from 'rapidfuzz-js/levenshtein'

distance('recieve', 'receive') // 2 — two single-character fixes apart
```

**A [Scorer](/concepts/scorers/) is a metric with decisions attached.** Real
applications make the same decisions on every comparison — what counts as
"close enough", how to treat missing values, how the algorithm is tuned.
`createScorer` locks those decisions into a reusable object so you make them
once.

**A [Matcher](/concepts/matchers/) applies a scorer to a whole collection.**
Give it your list of products, names, or records once; then ask it questions
— "what's the best match for this query?" — as many times as you like. It
does the expensive preparation up front so each query is fast.

For one-off questions there are shortcuts (`bestMatch`, `search`) that skip
the Matcher and just scan a collection directly.

## Do I need to understand the algorithms?

Not to get started. `weightedRatio` from `rapidfuzz-js/fuzz` picks a
sensible strategy automatically, and the
[Getting started](/getting-started/) page will have you matching in a
minute.

You'll want the [Algorithms](/algorithms/levenshtein/) section when results
surprise you — why "wireless keyboard" doesn't match "keyboard, wireless",
say — because each algorithm has a different opinion about what "similar"
means, and picking the right opinion for your data is most of the craft.
Each algorithm page ends with a plain "when to use it".

## Where to start

- [Getting started](/getting-started/) — install and first results.
- [Concepts](/concepts/metrics/) — the three building blocks, properly.
- [Guides](/guides/comparing-strings/) — task-by-task recipes.
- [Benchmarks](/benchmarks/) — how it compares to other JavaScript libraries.
