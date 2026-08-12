---
title: Errors
description: Every error the library throws, what caused it, and how to fix it.
---

The library prefers throwing to guessing. Almost every error here exists
because the alternative was a plausible wrong number — a silently ignored
option, a wrapped integer that still looks like a score, a threshold that
never applied.

Errors are grouped by what you were doing. Every message below is the exact
string thrown.

## Input validation

| Message                                              | Type        | Cause                                                            |
| ---------------------------------------------------- | ----------- | ---------------------------------------------------------------- |
| `expected a string or an array-like sequence`        | `TypeError` | A number, boolean, or object without a valid `length`            |
| `missing sequences are not supported by this scorer` | `TypeError` | `null`/`undefined` to a distance or a `missing: 'throw'` scorer  |
| `items must be a collection, not a single string`    | `TypeError` | A string passed where a collection of choices was expected       |
| `source item is missing`                             | `TypeError` | A missing item or `getText` result under `missingItems: 'throw'` |

_Missing_ means exactly `null` or `undefined`. Everything else that isn't a
sequence is **invalid**, and invalid always throws — a `NaN` or a `42` in
your data is a bug, not a gap. See
[Missing values](/concepts/scorers/#missing-values).

A single string is rejected as a collection because the alternative is worse:
it would silently search the string's characters and return single-letter
matches.

## Options and configuration

| Message                                           | Type         | Cause                                              |
| ------------------------------------------------- | ------------ | -------------------------------------------------- |
| `unknown <operation> option '<key>'`              | `TypeError`  | A misspelled option key                            |
| `<operation> options must be an object`           | `TypeError`  | A non-object in an options position                |
| `threshold must be finite`                        | `RangeError` | `Infinity`, `NaN`, or a misspelled `threshold` key |
| `unknown metric configuration key '<key>'`        | `TypeError`  | A key the built-in metric does not declare         |
| `unknown custom scorer configuration key '<key>'` | `TypeError`  | A stray key in a custom scorer's configuration     |
| `scorer was not created by createScorer`          | `TypeError`  | A hand-rolled object in a `scorer` slot            |

```ts
search('new york', teams, { scorer, thresold: 90 })
// TypeError: unknown search option 'thresold'

bestMatch('new york', teams, { scorer, limit: 2 })
// TypeError: unknown bestMatch option 'limit'
```

`<operation>` is the call you made: `search`, `bestMatch`, `searchIter`,
`scoreMatrix`, `scorePairs`, `prepareChoice`, or a `Matcher` method.

TypeScript's excess-property check only covers fresh object literals, so
without this runtime check a misspelled `threshold` would typecheck and
silently return unthresholded results. The `{ threshold }` argument to
`score`, `isMatch`, and `scoreIfMatch` is the exception — its one key is
required, so a misspelling already fails as `threshold must be finite`.

Configuration is checked against what the metric declares, so `weights` on
Jaro-Winkler, or `pad` on Levenshtein, throws rather than being ignored.
`missing` is similarity-only: passing it to a `distance` metric is
`unknown metric configuration key 'missing'`.

## Prepared choices

| Message                                                                   | Type        | Cause                                     |
| ------------------------------------------------------------------------- | ----------- | ----------------------------------------- |
| `prepared choice is incompatible with this scorer`                        | `TypeError` | A handle from a different scorer identity |
| `getPrepared returned an invalid prepared choice`                         | `TypeError` | Something that is not a handle at all     |
| `getPrepared cannot be combined with getText or missingItems`             | `TypeError` | Mixing prepared and unprepared mode       |
| `prepared choice was normalized, this search is not`                      | `TypeError` | Half-normalized prepared search           |
| `this search normalizes, the prepared choice was not`                     | `TypeError` | The other half                            |
| `prepared choice was normalized by a different function than this search` | `TypeError` | Two different normalizer functions        |

Normalization is all-or-nothing, and the handle records which side did it.
The last message is the one that surprises people: the check compares
**function identity**, so two arrow functions with the same body count as
different normalizers. Define one function and pass it to both sides. See
[Prepared choices](/guides/prepared-choices/).

## Batch scoring

| Message                                                         | Type         | Cause                                              |
| --------------------------------------------------------------- | ------------ | -------------------------------------------------- |
| `queries and choices must have the same length`                 | `RangeError` | `scorePairs` with mismatched inputs                |
| `(r, c) is outside a R × C matrix`                              | `RangeError` | `ScoreMatrix.at` out of range                      |
| `<operation> produced the score N, which '<kind>' cannot store` | `RangeError` | A batch score too wide for the `into` element type |

```ts
scoreMatrix(['cat'], ['cats'], { scorer, into: 'u8', scoreMultiplier: 3 })
// RangeError: scoreMatrix produced the score 257, which 'u8' cannot store:
// score into a wider element type, or into 'u8c' to saturate instead
```

An integer typed array wraps silently — `300` written into a `u8` reads back
as `44`, which still looks like a score. The check is skipped entirely where
the scorer's bounds and multiplier prove every score fits, so a `0–100`
scorer into `u8` costs nothing. `u8c` (`Uint8ClampedArray`) is the way to ask
for saturation on purpose and is exempt.

`at` throws rather than returning `undefined`, which is what keeps its return
type honestly `number`.

## Custom metrics

| Message                                                             | Type         | Cause                                                |
| ------------------------------------------------------------------- | ------------ | ---------------------------------------------------- |
| `custom metric returned a score outside its declared bounds`        | `RangeError` | A custom metric disagreeing with its own `bounds`    |
| `bounds must be an ordered numeric pair with a finite lower bound`  | `RangeError` | Reversed or non-finite custom bounds                 |
| `custom similarity bounds must include 0 unless missing is 'throw'` | `RangeError` | A `'compatible'` similarity whose bounds exclude `0` |

The metadata you declare is a contract, checked before any thresholding,
ordering, or pruning relies on it. The last rule follows from the first: a
`'compatible'` similarity returns `0` for a missing operand, so bounds that
exclude `0` would make the library violate its own check.

## Algorithm-specific

| Message                                          | Type         | Cause                                         |
| ------------------------------------------------ | ------------ | --------------------------------------------- |
| `Sequences are not the same length.`             | `Error`      | Hamming with `pad: false` and unequal lengths |
| `prefix_weight has to be in the range 0.0 - 1.0` | `RangeError` | Jaro-Winkler `prefixWeight` outside `[0, 1]`  |

These two keep RapidFuzz's own wording, including the trailing full stop and
the `snake_case` parameter name, so a message searched for in a Python
codebase's issue tracker still finds the same explanation.
