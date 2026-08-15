import { commonPrefix } from '../shared/affix.js'
import {
  convPair,
  asSequence,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type ScorerOptions,
  type MaybeSequence,
  prepareMetric,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MaybeSequenceMetricImplementation,
} from '../shared/scorerSupport.js'

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function preparedPrefixDistance(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return maximum(s1, s2) - commonPrefix(s1, s2)
}

function prefixDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return distCutoff(maximum(a, b) - commonPrefix(a, b), options.scoreCutoff)
}

function prefixSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return simCutoff(commonPrefix(a, b), options.scoreCutoff)
}

function prefixNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const max = maximum(a, b)
  return normDistCutoff(
    normalizeDistance(max - commonPrefix(a, b), max),
    options.scoreCutoff,
  )
}

function prefixNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const max = maximum(a, b)
  return normSimCutoff(
    1 - normalizeDistance(max - commonPrefix(a, b), max),
    options.scoreCutoff,
  )
}

export const prefixDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    prefixDistance_impl,
    DISTANCE_FLAGS,
    prepareMetric('distance', preparedPrefixDistance, maximum),
  )
export const prefixSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    prefixSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareMetric('similarity', preparedPrefixDistance, maximum),
  )
export const prefixNormalizedDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    prefixNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric('normalizedDistance', preparedPrefixDistance, maximum),
  )
export const prefixNormalizedSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    prefixNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedPrefixDistance, maximum),
  )
