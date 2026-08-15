import { builtInMetric, type BuiltInMetric } from '#core/scoring/builtIn/metric.js'

import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from './implementation.js'

/**
 * How many elements fall outside the common prefix, counted across both
 * inputs.
 *
 * ```ts
 * distance('apple', 'applesauce') // 5
 * ```
 */
export const distance: BuiltInMetric<'prefix.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixDistance,
    directImplementation: prefixDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * The length of the run of identical elements at the **start** of both
 * sequences.
 *
 * ```ts
 * similarity('apple', 'applesauce') // 5
 * ```
 *
 * Drastically cheaper than any edit distance, which makes it a good
 * pre-filter — but blind to anything after the first difference:
 * `similarity('apple', 'napple')` is `0`.
 */
export const similarity: BuiltInMetric<'prefix.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixSimilarity,
    directImplementation: prefixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the longer input.
 */
export const normalizedDistance: BuiltInMetric<'prefix.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixNormalizedDistance,
    directImplementation: prefixNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
/**
 * The shared prefix length over the longer input, `0..1`.
 *
 * ```ts
 * normalizedSimilarity('apple', 'applesauce') // 0.5
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'prefix.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: prefixNormalizedSimilarity,
  directImplementation: prefixNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
