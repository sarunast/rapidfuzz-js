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
  /**
   * What each element contributes to the overlap, keyed by the element itself.
   *
   * Only with `gramSize: 1`, where a gram *is* an element — a shingle of
   * several has no single weight to carry. `shared`, and each side's unmatched
   * remainder, become weighted masses instead of counts, so a generic token can
   * be priced below a distinctive one:
   *
   * ```ts
   * const company = createScorer(similarity, {
   *   gramSize: 1,
   *   elementWeights: new Map([
   *     ['swisscom', 5],
   *     ['ag', 0.1],
   *   ]),
   * })
   *
   * company.score(['swisscom', 'ag'], ['swisscom']) // 0.99 — `ag` costs little
   * company.score(['swisscom', 'ag'], ['ag']) // 0.0385 — the name is missing
   * ```
   *
   * Weights apply per **occurrence**, so `['react', 'react']` at `3` carries
   * `6`. An element the map does not name weighs `defaultElementWeight`, and
   * `0` drops one from the comparison entirely. Keys are canonical elements:
   * `'a'` and `97` are one element, and naming both with different weights is a
   * `RangeError` rather than a race between them.
   *
   * The map is **snapshotted** when the scorer is created — mutating it
   * afterwards changes nothing — and the trap is that weighting does not make
   * matching fuzzy: `swisscom` and `swisscomm` still share no mass at all.
   *
   * Weights that are all the same positive amount price nothing, since one
   * constant factor cancels from the ratio, and the scorer compiles to plain
   * unigram Tversky instead of paying for a weighted representation.
   *
   * @throws {TypeError} If it is present but not map-like, or any weight is not
   *   a number.
   * @throws {RangeError} If `gramSize` is not `1`, a weight is negative or not
   *   finite, one element is named twice with different weights, or the weights
   *   span a range too wide to represent.
   */
  readonly elementWeights?: ReadonlyMap<unknown, number> | undefined
  /**
   * What an element `elementWeights` does not name contributes, defaulting to
   * `1`.
   *
   * Set it to `0` to score only the elements named explicitly, which turns the
   * metric into overlap of a chosen vocabulary — and keeps the zero-mass rules
   * above, since an all-ignored pair is not an ordinary comparison.
   *
   * On its own — with no `elementWeights` — it still opts into weighted
   * configuration and still requires `gramSize: 1`. A positive value on its own
   * weighs every element alike, though, which prices nothing and therefore
   * compiles to ordinary unigram Tversky.
   *
   * @throws {TypeError} If it is present but not a number — `null` included.
   * @throws {RangeError} If it is negative, not finite, or `gramSize` is not
   *   `1`.
   */
  readonly defaultElementWeight?: number | undefined
}

/** {@link TverskyDistanceConfiguration} plus the missing-value policy. */
export interface TverskySimilarityConfiguration
  extends TverskyDistanceConfiguration, SimilarityConfiguration {}

const CONFIGURATION_KEYS: readonly string[] = [
  'gramSize',
  'alpha',
  'beta',
  'elementWeights',
  'defaultElementWeight',
]

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
