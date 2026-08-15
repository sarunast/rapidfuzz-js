import { builtInMetric, type BuiltInMetric } from '../../core/scoring/builtIn/metric.js'
import {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from './implementation.js'

/**
 * Levenshtein plus transposition: swapping two **adjacent** elements costs
 * one edit rather than two substitutions, which is what makes it fit
 * keyboard typos.
 *
 * ```ts
 * distance('recieve', 'receive') // 1 — one swap
 * ```
 *
 * Optimal string alignment restricts what Damerau-Levenshtein allows: once
 * a stretch has been edited it cannot be edited again, so
 * `distance('ca', 'abc')` is `3` where the unrestricted algorithm finds `2`.
 * On realistic typo data the two agree almost always.
 */
export const distance: BuiltInMetric<'osa.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * How much the two sequences share, in edit units: `maximum − distance`.
 *
 * **Not a 0–1 score** — {@link normalizedSimilarity} is the fraction.
 */
export const similarity: BuiltInMetric<'osa.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the longer input.
 */
export const normalizedDistance: BuiltInMetric<'osa.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
/**
 * {@link similarity} as a `0..1` fraction of the longer input — `1` identical.
 */
export const normalizedSimilarity: BuiltInMetric<
  'osa.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: osaNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
