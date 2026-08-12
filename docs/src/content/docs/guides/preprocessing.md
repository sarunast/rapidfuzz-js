---
title: Preprocessing
description: Why "Apple!" scores badly against "apple", and how normalization fixes it before scoring.
---

To an edit-distance algorithm, `'Apple!'` and `'apple'` differ in **two**
places — the capital A and the `!` are both edits, the same as real typos.
Case, punctuation, and stray whitespace are the most common reason fuzzy
scores come out mysteriously low.

Preprocessing fixes this at the root: transform both strings into a clean,
comparable form _before_ scoring, so only meaningful differences count.

## The built-in: normalizeText

```ts
import { normalizeText } from 'rapidfuzz-js'

normalizeText('  Wireless-Mechanical KEYBOARD!! ')
// 'wireless mechanical keyboard'
```

Precisely: every character that isn't a Unicode letter, number, or `_`
becomes a space; the result is trimmed and lowercased. Runs of separators
aren't collapsed — `'a---b'` becomes `'a   b'` — and no Unicode `NFC`/`NFKC`
form is applied.

Non-string sequences pass through untouched, which is what lets it stay the
normalizer for a collection of array-like choices where no element is text
to lowercase. Values that aren't sequences at all — numbers, booleans —
throw a `TypeError`.

This is the port of RapidFuzz's `utils.default_process`, and like upstream
it's **opt-in** — nothing applies it unless you ask. Scoring exactly what
you were given is the only honest default; cleaning is a decision about
your data.

## Normalizing in a search

`bestMatch`, `search`, and `createMatcher` all take `normalize`. The crucial
property: it's applied to **both sides** — every item at preparation time
and every query at call time — so the comparison always happens between two
cleaned strings:

```ts
import { createMatcher, createScorer, normalizeText } from 'rapidfuzz-js'
import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortSimilarity)

const matcher = createMatcher(['Wireless-Mechanical KEYBOARD'], {
  scorer,
  normalize: normalizeText,
})

matcher.best('wireless keyboard') // matches cleanly
```

Normalizing only one side is the classic mistake — never do it by hand on
the query while the collection stays raw. The `normalize` option exists so
the two sides can't drift apart.

## Writing your own

`normalizeText` is deliberately blunt — it erases _all_ punctuation and
case. When that's too much (or not enough — accents, unicode forms, domain
noise like `"Ltd."`), a normalizer is just a function
`(value) => cleaned value`:

```ts
const normalize = (value) =>
  typeof value === 'string' ? value.normalize('NFKC').toLowerCase() : value
```

Two rules:

- **Deterministic.** A Matcher normalizes items once, at construction; a
  normalizer that answers differently later would leave queries compared
  against stale text.
- **Returning `null`/`undefined` means "nothing to search"** — the item is
  treated as missing and [skipped](/concepts/matchers/#gaps-in-the-data),
  which doubles as a filter for items that shouldn't be searchable.

## Objects: getText runs first

For collections of objects, `getText` extracts the text, then `normalize`
cleans it:

```ts
const matcher = createMatcher(products, {
  scorer,
  getText: (product) => product.title,
  normalize: normalizeText,
})
```

Results always return your original items untouched — preprocessing changes
what gets _scored_, never what you get _back_.

## Direct metric calls

Bare metrics and scorers apply no preprocessing at all. Comparing pairs by
hand? Normalize both sides yourself:

```ts
similarity(normalizeText(a), normalizeText(b))
```
