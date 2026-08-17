import type { MaybeSequenceMetricImplementation } from '#core/scoring/builtIn/implementation.js'
import {
  builtInMetric,
  type ExplainableBuiltInMetric,
} from '#core/scoring/builtIn/metric.js'
import type { Metric } from '#core/scoring/metric.js'
import type { Direction, SimilarityConfiguration } from '#core/types.js'

import type { TverskyElementSimilarity } from './elementSimilarity.js'
import type { TverskyEvidence } from './evidence.js'
import {
  tverskyDistance,
  tverskySimilarity,
  type TverskyOptions,
} from './implementation.js'

export type { TverskyElementSimilarity } from './elementSimilarity.js'
export type {
  TverskyEvidence,
  TverskyEvidenceMatch,
  TverskyEvidenceTotals,
  TverskyUnmatchedElement,
} from './evidence.js'

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
  /**
   * Let elements that are not equal still share mass, by scoring the leftovers
   * of exact matching with an inner element scorer.
   *
   * This is what makes `['swisscom', 'ag']` and `['swisscomm', 'ag']` score
   * close to `1` instead of counting `swisscom` as missing on both sides. Only
   * multi-character string tokens are compared — see
   * {@link TverskyElementSimilarity} for that and the other four traps.
   *
   * @throws {TypeError} If it is not an object, holds an unknown key, or its
   *   `scorer` is not a symmetric similarity scorer from `createScorer`.
   * @throws {RangeError} If `gramSize` is not `1`, `threshold` is outside
   *   `0 < threshold <= 1`, or the scorer's bounds do not span a finite,
   *   non-zero range. Scoring throws one too, where a pair leaves more than 32
   *   distinct fuzzy-comparable leftovers on a side, or its occurrence counts are
   *   skewed enough to need more than 512 augmenting paths to match.
   */
  readonly elementSimilarity?: TverskyElementSimilarity | undefined
}

/** {@link TverskyDistanceConfiguration} plus the missing-value policy. */
export interface TverskySimilarityConfiguration
  extends TverskyDistanceConfiguration, SimilarityConfiguration {}

/**
 * The configuration that also buys `scorer.explain(first, second)`.
 *
 * Exact element overlap — `gramSize: 1` — is the only shape with occurrences a
 * caller named, so it is the only one that can say which element matched which
 * and what each cost. A shingle of several elements is not one thing anybody
 * asked about, weighted or not.
 *
 * `createScorer` reads the configuration *literal*, so hoisting one into a
 * variable widens `gramSize` to `number` and quietly gives back an ordinary
 * scorer. Name this type to keep it:
 *
 * ```ts
 * const config = { gramSize: 1, alpha: 1, beta: 0.1 } satisfies TverskyExplainConfiguration
 * const company = createScorer(similarity, config) // still explains
 * ```
 *
 * This one carries no `missing`, since a distance metric refuses it — hoist a
 * similarity configuration as {@link TverskySimilarityExplainConfiguration}
 * instead. Written inline, either metric accepts what it accepts.
 */
export interface TverskyExplainConfiguration extends TverskyDistanceConfiguration {
  /** Exactly `1`: one element to a gram, so an occurrence is a whole element. */
  readonly gramSize: 1
}

/**
 * {@link TverskyExplainConfiguration} plus the missing-value policy: what a
 * hoisted configuration for {@link similarity} is named.
 *
 * It stands to {@link TverskyExplainConfiguration} exactly as
 * {@link TverskySimilarityConfiguration} stands to
 * {@link TverskyDistanceConfiguration} — `missing` is accepted by a similarity
 * and refused by a distance, so the explanation configurations divide the same
 * way.
 *
 * ```ts
 * const config = {
 *   gramSize: 1,
 *   missing: 'throw',
 * } satisfies TverskySimilarityExplainConfiguration
 * const tokens = createScorer(similarity, config) // still explains
 * ```
 */
export interface TverskySimilarityExplainConfiguration
  extends TverskyExplainConfiguration, SimilarityConfiguration {}

const CONFIGURATION_KEYS: readonly string[] = [
  'gramSize',
  'alpha',
  'beta',
  'elementWeights',
  'defaultElementWeight',
  'elementSimilarity',
]

function tverskyMetric<
  TDirection extends Direction,
  TConfig extends object,
  TBrand,
  TExplains extends TConfig,
>(
  implementation: MaybeSequenceMetricImplementation<TverskyOptions, TverskyEvidence>,
  direction: TDirection,
): Metric<TDirection, TConfig, TBrand, TExplains, TverskyEvidence> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: CONFIGURATION_KEYS,
  })
}

/** `1 − similarity`, on the same `0..1` scale. */
export const distance: ExplainableBuiltInMetric<
  'tversky.distance',
  'distance',
  TverskyDistanceConfiguration,
  TverskyExplainConfiguration,
  TverskyEvidence
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
 *
 * At `gramSize: 1` a scorer also gains `explain(first, second)`, which reports
 * the {@link TverskyEvidence} behind a score — see
 * {@link TverskyExplainConfiguration}.
 */
export const similarity: ExplainableBuiltInMetric<
  'tversky.similarity',
  'similarity',
  TverskyDistanceConfiguration,
  TverskyExplainConfiguration,
  TverskyEvidence
> = /* @__PURE__ */ tverskyMetric(tverskySimilarity, 'similarity')

/** Tversky is already `0..1`, so this is {@link distance} itself. */
export const normalizedDistance: typeof distance = distance
/** Tversky is already `0..1`, so this is {@link similarity} itself. */
export const normalizedSimilarity: typeof similarity = similarity
