import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { jaroWinklerDistance, jaroWinklerSimilarity } from './implementation.js'

/** {@link JaroWinklerDistanceConfiguration} plus the missing-value policy. */
export interface JaroWinklerConfiguration extends SimilarityConfiguration {
  /** See {@link JaroWinklerDistanceConfiguration.prefixWeight}. */
  readonly prefixWeight?: number | undefined
}

/** Accepted by every Jaro-Winkler metric. */
export interface JaroWinklerDistanceConfiguration {
  /**
   * How much each shared leading element is worth, defaulting to `0.1`. Each of
   * the first four adds `prefixWeight × (1 − jaroScore)` to the score.
   *
   * Values above `0.25` would let a score exceed `1`.
   */
  readonly prefixWeight?: number | undefined
}

const PREFIX_WEIGHT: readonly string[] = ['prefixWeight']

function jaroWinklerMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation,
  direction: TDirection,
): Metric<TDirection, TConfig, TBrand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: PREFIX_WEIGHT,
  })
}

/** `1 − similarity`, on the same `0..1` scale. */
export const distance: BuiltInMetric<
  'jaroWinkler.distance',
  'distance',
  JaroWinklerDistanceConfiguration
> = /* @__PURE__ */ jaroWinklerMetric(jaroWinklerDistance, 'distance')
/**
 * Jaro with a bonus for a shared prefix, `0..1` — the de-facto standard for
 * matching people and place names.
 *
 * ```ts
 * similarity('martha', 'marhta') // 0.9611… — Jaro's 0.9444 plus the 'mar' bonus
 * ```
 *
 * The bias is also the limitation: strings differing at the very start score
 * noticeably worse, so for typos that land anywhere prefer Levenshtein or OSA.
 * Tune the bonus with `prefixWeight`.
 */
export const similarity: BuiltInMetric<
  'jaroWinkler.similarity',
  'similarity',
  JaroWinklerDistanceConfiguration
> = /* @__PURE__ */ jaroWinklerMetric(jaroWinklerSimilarity, 'similarity')

// Jaro-Winkler is normalized by construction, so these are the same metrics
// under the names the other algorithms use. `typeof` carries the identity
// across instead of restating it, which is what keeps their prepared choices
// interchangeable.
/** Jaro-Winkler is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Jaro-Winkler is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
