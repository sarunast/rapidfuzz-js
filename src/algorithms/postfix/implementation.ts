import { commonSuffix } from '../shared/affix.js'
import {
  asSequence,
  convPair,
  distCutoff,
  normalize,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type MaybeSequence,
  type NormalizedScorer,
  type ScorerOptions,
  prepareMetric,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
} from '../shared/scorerSupport.js'

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function preparedPostfixDistance(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return maximum(s1, s2) - commonSuffix(s1, s2)
}

/**
 * Elements outside the common suffix: `max(|s1|, |s2|)` minus the length of the
 * longest common suffix.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
function postfixDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return distCutoff(maximum(a, b) - commonSuffix(a, b), options.scoreCutoff)
}

function postfixSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return simCutoff(commonSuffix(a, b), options.scoreCutoff)
}

function postfixNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const max = maximum(a, b)
  return normDistCutoff(normalize(max - commonSuffix(a, b), max), options.scoreCutoff)
}

/**
 * Postfix similarity normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function postfixNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const max = maximum(a, b)
  return normSimCutoff(1 - normalize(max - commonSuffix(a, b), max), options.scoreCutoff)
}

export const postfixDistance: NormalizedScorer = /* @__PURE__ */ withPreparedFlags(
  postfixDistance_impl,
  DISTANCE_FLAGS,
  prepareMetric('distance', preparedPostfixDistance, maximum),
)
export const postfixSimilarity: NormalizedScorer = /* @__PURE__ */ withPreparedFlags(
  postfixSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareMetric('similarity', preparedPostfixDistance, maximum),
)
export const postfixNormalizedDistance: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric('normalizedDistance', preparedPostfixDistance, maximum),
  )
export const postfixNormalizedSimilarity: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedPostfixDistance, maximum),
  )
