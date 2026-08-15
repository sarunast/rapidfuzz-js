import type { MaybeSequenceMetricImplementation } from '#core/scoring/builtIn/implementation.js'
import { builtInMetric, type BuiltInMetric } from '#core/scoring/builtIn/metric.js'
import type { Metric } from '#core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '#core/types.js'

import {
  hammingDistance,
  hammingEditops,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingOpcodes,
  hammingSimilarity,
  type HammingOptions,
} from './implementation.js'

/** Accepted by every Hamming metric. */
export interface HammingDistanceConfiguration {
  /**
   * Whether a length difference is counted as differences rather than refused.
   * Defaults to `true`.
   *
   * With `pad: false` the metric throws on unequal lengths, which is the
   * stricter reading for fixed-width data where a length difference means the
   * input is wrong rather than merely different.
   */
  readonly pad?: boolean | undefined
}
/** {@link HammingDistanceConfiguration} plus the missing-value policy. */
export interface HammingSimilarityConfiguration
  extends HammingDistanceConfiguration, SimilarityConfiguration {}

const PAD: readonly string[] = ['pad']

function hammingMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<HammingOptions>,
  direction: TDirection,
  bounds: readonly [number, number],
): Metric<TDirection, TConfig, TBrand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds,
    configurationKeys: PAD,
  })
}

/**
 * How many positions differ, comparing the two sequences in lockstep.
 * Nothing shifts and nothing aligns — position 3 is compared with
 * position 3.
 *
 * ```ts
 * distance('karolin', 'kathrin') // 3
 * ```
 *
 * By default the shorter input is treated as padded, so a length
 * difference counts as differences. Set `pad: false` to reject unequal
 * lengths instead.
 *
 * @throws `Error` when `pad` is `false` and the lengths differ.
 */
export const distance: BuiltInMetric<
  'hamming.distance',
  'distance',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingDistance, 'distance', [
  0,
  Number.POSITIVE_INFINITY,
])
/**
 * How many positions match — `maximum − distance`.
 *
 * **Not a 0–1 score** — {@link normalizedSimilarity} is the fraction.
 */
export const similarity: BuiltInMetric<
  'hamming.similarity',
  'similarity',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingSimilarity, 'similarity', [
  0,
  Number.POSITIVE_INFINITY,
])
/**
 * {@link distance} as a `0..1` fraction of the compared length.
 */
export const normalizedDistance: BuiltInMetric<
  'hamming.normalizedDistance',
  'distance',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingNormalizedDistance, 'distance', [0, 1])
/**
 * The share of positions that match, `0..1`.
 *
 * ```ts
 * normalizedSimilarity('karolin', 'kathrin') // 0.5714…
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'hamming.normalizedSimilarity',
  'similarity',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingNormalizedSimilarity, 'similarity', [0, 1])
export { hammingEditops as editops, hammingOpcodes as opcodes }
