import { commonSuffix } from '../shared/affix.js'
import {
  convPair,
  distCutoff,
  normalize,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  prepareMetric,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  type Scorer,
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
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  return distCutoff(maximum(a, b) - commonSuffix(a, b), options.scoreCutoff)
}

/**
 * Postfix similarity normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function postfixNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  return normSimCutoff(1 - normalize(max - commonSuffix(a, b), max), options.scoreCutoff)
}

export const postfixDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  postfixDistance_impl,
  DISTANCE_FLAGS,
  prepareMetric('distance', preparedPostfixDistance, maximum),
)
export const postfixNormalizedSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  postfixNormalizedSimilarity_impl,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareMetric('normalizedSimilarity', preparedPostfixDistance, maximum),
)
