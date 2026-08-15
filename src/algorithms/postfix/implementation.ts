import {
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
} from '#core/scoring/builtIn/cutoff.js'
import {
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
} from '#core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '#core/scoring/builtIn/options.js'
import { prepareMetric } from '#core/scoring/builtIn/preparation.js'
import { validateSequence, convPair, maxSequenceLength } from '#core/sequence.js'
import type { MaybeSequence } from '#core/types.js'

import { commonSuffix } from '../affix.js'

function preparedPostfixDistance(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return maxSequenceLength(s1, s2) - commonSuffix(s1, s2)
}

function postfixDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return distCutoff(maxSequenceLength(a, b) - commonSuffix(a, b), options.scoreCutoff)
}

function postfixSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return simCutoff(commonSuffix(a, b), options.scoreCutoff)
}

function postfixNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const max = maxSequenceLength(a, b)
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
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const max = maxSequenceLength(a, b)
  return normSimCutoff(
    1 - normalizeDistance(max - commonSuffix(a, b), max),
    options.scoreCutoff,
  )
}

export const postfixDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixDistance_impl,
    DISTANCE_FLAGS,
    prepareMetric('distance', preparedPostfixDistance, maxSequenceLength),
  )
export const postfixSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareMetric('similarity', preparedPostfixDistance, maxSequenceLength),
  )
export const postfixNormalizedDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric('normalizedDistance', preparedPostfixDistance, maxSequenceLength),
  )
export const postfixNormalizedSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    postfixNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedPostfixDistance, maxSequenceLength),
  )
