import type { Metric } from '../../core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { cosineDistance, cosineSimilarity, type CosineOptions } from './implementation.js'

/** Accepted by every Cosine metric. */
export interface CosineDistanceConfiguration {
  /**
   * How many adjacent elements make one gram, defaulting to `2`.
   *
   * Larger grams demand longer runs of exact agreement: at `3`,
   * `('banana', 'bananas')` eases from `0.9486…` to `0.9258…`, while
   * `('night', 'nacht')` falls from `0.25` to `0`. Inputs shorter than one gram
   * have no grams at all, and score `1` against an equal one and `0` against
   * anything else.
   *
   * It is also the whole of a prepared choice's identity — a scorer left at the
   * default and one written as `{ gramSize: 2 }` accept each other's handles.
   *
   * @throws {RangeError} If it is below `1` or not a safe integer.
   */
  readonly gramSize?: number | undefined
}

/** {@link CosineDistanceConfiguration} plus the missing-value policy. */
export interface CosineSimilarityConfiguration
  extends CosineDistanceConfiguration, SimilarityConfiguration {}

const GRAM_SIZE: readonly string[] = ['gramSize']

function cosineMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<CosineOptions>,
  direction: TDirection,
): Metric<TDirection, TConfig, TBrand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: GRAM_SIZE,
  })
}

/** `1 − similarity`, on the same `0..1` scale. */
export const distance: BuiltInMetric<
  'cosine.distance',
  'distance',
  CosineDistanceConfiguration
> = /* @__PURE__ */ cosineMetric(cosineDistance, 'distance')
/**
 * The angle between two n-gram frequency vectors, `0..1` — `dice.similarity`'s
 * sibling over the same bags, weighting repeats far more heavily.
 *
 * `Σ aᵍ · bᵍ / (‖A‖ · ‖B‖)` over bags of `gramSize` adjacent elements, with
 * nothing padded onto either end.
 *
 * ```ts
 * similarity('night', 'nacht') // 0.25 — one shared gram, `ht`
 * similarity('ababab', 'abab') // 0.9922… — `dice.similarity`: 0.75
 * ```
 *
 * Two things surprise people arriving from elsewhere. This is the dot product
 * of the frequency vectors, not the intersection-count formula
 * `|A ∩ B| / √(|A| · |B|)` — Otsuka-Ochiai — that several packages publish
 * under the name; the two always agree when no gram repeats, and can part
 * company once one does. And a repeated gram dominates, as the second line
 * above shows.
 *
 * Prefer `dice.similarity` when scoring a query against many candidates: Dice
 * bounds its score from gram counts and rejects a hopeless candidate without
 * building a profile, where this builds both every time.
 */
export const similarity: BuiltInMetric<
  'cosine.similarity',
  'similarity',
  CosineDistanceConfiguration
> = /* @__PURE__ */ cosineMetric(cosineSimilarity, 'similarity')

// Cosine is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
/** Cosine is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Cosine is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
