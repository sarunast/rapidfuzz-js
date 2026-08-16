import type { MaybeSequenceMetricImplementation } from '#core/scoring/builtIn/implementation.js'
import { builtInMetric, type BuiltInMetric } from '#core/scoring/builtIn/metric.js'
import type { Metric } from '#core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '#core/types.js'

import {
  tverskyDistance,
  tverskySimilarity,
  type TverskyOptions,
} from './implementation.js'

/** Accepted by every Tversky metric. */
export interface TverskyDistanceConfiguration {
  /**
   * How many adjacent elements make one gram, defaulting to `2`.
   *
   * Larger grams demand longer runs of exact agreement, and `1` turns the
   * metric into plain element overlap — hand it token arrays instead of
   * strings and it scores exact-token overlap with no substring credit at
   * all. Inputs shorter than one gram have no grams, and score `1` against an
   * equal input and `0` against anything else.
   *
   * @throws {TypeError} If it is present but not a number.
   * @throws {RangeError} If it is below `1` or not a safe integer.
   */
  readonly gramSize?: number | undefined
  /**
   * How much each gram found only in the **first** sequence costs, defaulting
   * to `0.5`.
   *
   * Any finite non-negative number is accepted, though not `0` while `beta`
   * is also `0`. The weights change scoring only — the profile a choice is
   * prepared into depends on `gramSize` alone — but prepared handles are
   * owned per configured scorer: only the full default configuration —
   * `gramSize` `2`, `alpha` and `beta` `0.5` — shares them with an
   * unconfigured scorer.
   *
   * @throws {TypeError} If it is present but not a number — `null` included.
   * @throws {RangeError} If it is negative, not finite, or `0` while `beta`
   *   is `0`.
   */
  readonly alpha?: number | undefined
  /**
   * How much each gram found only in the **second** sequence costs,
   * defaulting to `0.5`.
   *
   * The mirror of `alpha`: lowering it forgives extra content in the second
   * sequence, which is what makes `{ alpha: 1, beta: 0 }` ask "how completely
   * does the second sequence contain the first?".
   *
   * @throws {TypeError} If it is present but not a number — `null` included.
   * @throws {RangeError} If it is negative, not finite, or `0` while `alpha`
   *   is `0`.
   */
  readonly beta?: number | undefined
}

/** {@link TverskyDistanceConfiguration} plus the missing-value policy. */
export interface TverskySimilarityConfiguration
  extends TverskyDistanceConfiguration, SimilarityConfiguration {}

const CONFIGURATION_KEYS: readonly string[] = ['gramSize', 'alpha', 'beta']

function tverskyMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<TverskyOptions>,
  direction: TDirection,
): Metric<TDirection, TConfig, TBrand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: CONFIGURATION_KEYS,
  })
}

/** `1 − similarity`, on the same `0..1` scale. */
export const distance: BuiltInMetric<
  'tversky.distance',
  'distance',
  TverskyDistanceConfiguration
> = /* @__PURE__ */ tverskyMetric(tverskyDistance, 'distance')
/**
 * N-gram overlap with a separate price on each side's unmatched grams,
 * `0..1` — position ignored entirely.
 *
 * `shared / (shared + α·firstOnly + β·secondOnly)` over bags of `gramSize`
 * adjacent elements, counting repeats, with nothing padded onto either end.
 * `alpha` prices grams only the first sequence has, `beta` grams only the
 * second has — the defaults of `0.5` each make it exactly `dice.similarity`,
 * `{ alpha: 1, beta: 1 }` is multiset Jaccard, and `{ alpha: 1, beta: 0 }`
 * measures how completely the second sequence contains the first. Those are
 * equivalences of one formula, not separate modes.
 *
 * ```ts
 * import { createScorer } from 'rapidfuzz-js'
 * import { similarity } from 'rapidfuzz-js/tversky'
 *
 * similarity('night', 'nacht') // 0.25 — the Dice default
 *
 * const containment = createScorer(similarity, { alpha: 1, beta: 0 })
 * containment.score('bana', 'banana') // 1 — every query bigram is covered
 * containment.score('banana', 'bana') // 0.6 — two query bigrams are not
 * ```
 *
 * The trap follows from that example: once `alpha` and `beta` differ, the
 * metric is asymmetric, so swapping the arguments changes the score — keep
 * the query first. And containment is generous by construction:
 * `{ alpha: 1, beta: 0 }` scores a flat `1` for *any* second sequence that
 * covers the first's grams, however much else it carries.
 */
export const similarity: BuiltInMetric<
  'tversky.similarity',
  'similarity',
  TverskyDistanceConfiguration
> = /* @__PURE__ */ tverskyMetric(tverskySimilarity, 'similarity')

/** Tversky is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Tversky is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
