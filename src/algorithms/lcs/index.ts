import { builtInMetric, type BuiltInMetric } from '#core/scoring/builtIn/metric.js'

import {
  lcsSeqDistance,
  lcsSeqEditops,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqOpcodes,
  lcsSeqSimilarity,
} from './implementation.js'

/**
 * How many elements fall outside the longest common subsequence — the
 * characters that would have to be dropped from both sides.
 *
 * ```ts
 * distance('ABCBDAB', 'BDCABA') // 3
 * ```
 */
export const distance: BuiltInMetric<'lcs.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * The length of the longest common subsequence itself — the longest run of
 * elements appearing in both sequences in the same order, not necessarily
 * adjacent.
 *
 * ```ts
 * similarity('ABCBDAB', 'BDCABA') // 4
 * ```
 *
 * The one place in the library where the raw `similarity` is the headline
 * number rather than a by-product.
 */
export const similarity: BuiltInMetric<'lcs.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
/**
 * {@link distance} as a `0..1` fraction of the longer input.
 */
export const normalizedDistance: BuiltInMetric<'lcs.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
/**
 * The common subsequence length over the longer input, `0..1`.
 *
 * ```ts
 * normalizedSimilarity('ABCBDAB', 'BDCABA') // 0.5714…
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'lcs.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: lcsSeqNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
export { lcsSeqEditops as editops, lcsSeqOpcodes as opcodes }
