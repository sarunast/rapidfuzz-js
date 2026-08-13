import type { Metric } from '../../core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MetricImplementation } from '../shared/scorerSupport.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
  type LevenshteinWeightsInput,
} from './metric.js'
import type { LevenshteinOptions } from './types.js'

/** Accepted by every Levenshtein metric. */
export interface LevenshteinDistanceConfiguration {
  /**
   * What each operation costs. Defaults to `1` apiece; accepts
   * `{ insertion, deletion, substitution }` or the tuple
   * `[insertion, deletion, substitution]`.
   *
   * Substitution at `2` is Indel distance, which has its own faster subpath.
   */
  readonly weights?: LevenshteinWeightsInput | undefined
}

/** {@link LevenshteinDistanceConfiguration} plus the missing-value policy. */
export interface LevenshteinSimilarityConfiguration
  extends LevenshteinDistanceConfiguration, SimilarityConfiguration {}

const WEIGHTS: readonly string[] = ['weights']

function levenshteinMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MetricImplementation<LevenshteinOptions>,
  direction: TDirection,
  bounds: readonly [number, number],
): Metric<TDirection, TConfig, TBrand> {
  return builtInMetric({
    implementation,
    direction,
    bounds,
    configurationKeys: WEIGHTS,
  })
}

/**
 * How many single-character edits turn one sequence into the other — insert,
 * delete or substitute, each costing `1` by default.
 *
 * ```ts
 * distance('kitten', 'sitting') // 3
 * ```
 */
export const distance: BuiltInMetric<
  'levenshtein.distance',
  'distance',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinDistance, 'distance', [
  0,
  Number.POSITIVE_INFINITY,
])
/**
 * How much the two sequences share, in edit units: `maximum − distance`.
 *
 * **Not a 0–1 score.** `similarity('kitten', 'sitting')` is `4`, not `0.571` —
 * {@link normalizedSimilarity} is the fraction.
 */
export const similarity: BuiltInMetric<
  'levenshtein.similarity',
  'similarity',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinSimilarity, 'similarity', [
  0,
  Number.POSITIVE_INFINITY,
])
/**
 * {@link distance} as a `0..1` fraction of the longer input — `0` identical,
 * `1` nothing in common. Use it to compare across inputs of different lengths,
 * where three edits is bad for a word and trivial for a paragraph.
 */
export const normalizedDistance: BuiltInMetric<
  'levenshtein.normalizedDistance',
  'distance',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinNormalizedDistance, 'distance', [0, 1])
/**
 * {@link similarity} as a `0..1` fraction of the longer input — `1` identical.
 *
 * ```ts
 * normalizedSimilarity('kitten', 'sitting') // 0.5714…
 * ```
 */
export const normalizedSimilarity: BuiltInMetric<
  'levenshtein.normalizedSimilarity',
  'similarity',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(
  levenshteinNormalizedSimilarity,
  'similarity',
  [0, 1],
)
