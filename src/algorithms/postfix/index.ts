import { builtInMetric, type BuiltInMetric } from '../../core/scoring/builtIn/metric.js'
import {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from './implementation.js'

/**
 * How many elements fall outside the common suffix, counted across both
 * inputs.
 */
export const distance: BuiltInMetric<'postfix.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixDistance,
    directImplementation: postfixDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * The length of the run of identical elements at the **end** of both
 * sequences.
 *
 * ```ts
 * similarity('walking', 'running') // 3 — the shared 'ing'
 * ```
 */
export const similarity: BuiltInMetric<'postfix.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixSimilarity,
    directImplementation: postfixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the longer input.
 */
export const normalizedDistance: BuiltInMetric<'postfix.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixNormalizedDistance,
    directImplementation: postfixNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
/**
 * The shared suffix length over the longer input, `0..1`.
 *
 * ```ts
 * normalizedSimilarity('walking', 'running') // 0.4285…
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'postfix.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: postfixNormalizedSimilarity,
  directImplementation: postfixNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
