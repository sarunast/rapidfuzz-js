import { builtInMetric, type BuiltInMetric } from '#core/scoring/builtIn/metric.js'

import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
  damerauLevenshteinSimilarity,
} from './implementation.js'

/**
 * The unrestricted four-operation edit distance — insert, delete,
 * substitute, and transpose adjacent elements — with no limit on how the
 * operations combine.
 *
 * ```ts
 * distance('ca', 'abc') // 2 — swap to 'ac', then insert 'b'
 * ```
 *
 * Choose it over OSA when you need a true
 * metric (its distances satisfy the triangle inequality, OSA's do not) or
 * number-for-number agreement with another implementation's
 * "Damerau-Levenshtein". Otherwise OSA is the cheaper algorithm.
 */
export const distance: BuiltInMetric<'damerauLevenshtein.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: damerauLevenshteinDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * How much the two sequences share, in edit units: `maximum − distance`.
 *
 * **Not a 0–1 score** — {@link normalizedSimilarity} is the fraction.
 */
export const similarity: BuiltInMetric<'damerauLevenshtein.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: damerauLevenshteinSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the longer input.
 */
export const normalizedDistance: BuiltInMetric<
  'damerauLevenshtein.normalizedDistance',
  'distance'
> = /* @__PURE__ */ builtInMetric({
  implementation: damerauLevenshteinNormalizedDistance,
  direction: 'distance',
  bounds: [0, 1],
})
/**
 * {@link similarity} as a `0..1` fraction of the longer input — `1` identical.
 *
 * ```ts
 * normalizedSimilarity('ca', 'abc') // 0.333…
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'damerauLevenshtein.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: damerauLevenshteinNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
