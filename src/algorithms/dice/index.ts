import type { Metric } from '../../core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { diceDistance, diceSimilarity, type DiceOptions } from './implementation.js'

/** Accepted by every Sørensen-Dice metric. */
export interface DiceDistanceConfiguration {
  /**
   * How many adjacent elements make one gram, defaulting to `2`.
   *
   * Larger grams demand longer runs of exact agreement: at `3`,
   * `('night', 'nacht')` falls from `0.25` to `0`, while
   * `('banana', 'bananas')` only eases from `0.9090…` to `0.8888…`. Inputs
   * shorter than one gram have no grams at all, and score `1` against an equal
   * one and `0` against anything else.
   *
   * It is also the whole of a prepared choice's identity — a scorer left at the
   * default and one written as `{ gramSize: 2 }` accept each other's handles.
   *
   * @throws {RangeError} If it is below `1` or not a safe integer.
   */
  readonly gramSize?: number | undefined
}

/** {@link DiceDistanceConfiguration} plus the missing-value policy. */
export interface DiceSimilarityConfiguration
  extends DiceDistanceConfiguration, SimilarityConfiguration {}

const GRAM_SIZE: readonly string[] = ['gramSize']

function diceMetric<TDirection extends Direction, TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<DiceOptions>,
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
  'dice.distance',
  'distance',
  DiceDistanceConfiguration
> = /* @__PURE__ */ diceMetric(diceDistance, 'distance')
/**
 * How much of two sequences' n-grams overlap, `0..1` — position ignored
 * entirely.
 *
 * `2 × shared / (gramsLeft + gramsRight)` over bags of `gramSize` adjacent
 * elements, counting repeats, with nothing padded onto either end.
 *
 * ```ts
 * similarity('night', 'nacht') // 0.25 — `ni ig gh ht` against `na ac ch ht`
 * similarity('new york mets', 'mets new york') // 0.8333… — Levenshtein: 0.2307…
 * ```
 *
 * That second line is the reason to reach for it, and the trap is its mirror:
 * order is gone, so `('aba', 'bab')` is a flat `1`. A plain typo also costs
 * more than its size — `('recieve', 'receive')` is `0.5` where Levenshtein
 * `normalizedSimilarity` says `0.7142…`.
 *
 * Under a threshold this is the cheaper of the two n-gram metrics: gram counts
 * alone bound the score, so a candidate whose length rules it out is rejected
 * before either bag is built. `cosine.similarity` has no such bound.
 */
export const similarity: BuiltInMetric<
  'dice.similarity',
  'similarity',
  DiceDistanceConfiguration
> = /* @__PURE__ */ diceMetric(diceSimilarity, 'similarity')

// Dice is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
/** Dice is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Dice is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
