import { commonSuffix } from '../shared/affix.js'
import {
  asSequence,
  convPair,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type MaybeSequence,
  type MaybeSequenceMetricImplementation,
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
  return normDistCutoff(
    normalizeDistance(max - commonSuffix(a, b), max),
    options.scoreCutoff,
  )
}

function postfixNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const max = maximum(a, b)
  return normSimCutoff(
    1 - normalizeDistance(max - commonSuffix(a, b), max),
    options.scoreCutoff,
  )
}

export const postfixDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixDistance_impl,
    DISTANCE_FLAGS,
    prepareMetric('distance', preparedPostfixDistance, maximum),
  )
export const postfixSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareMetric('similarity', preparedPostfixDistance, maximum),
  )
export const postfixNormalizedDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric('normalizedDistance', preparedPostfixDistance, maximum),
  )
export const postfixNormalizedSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedPostfixDistance, maximum),
  )
