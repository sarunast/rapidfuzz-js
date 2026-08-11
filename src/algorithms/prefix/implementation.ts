import { commonPrefix } from '../shared/affix.js'
import {
  conv,
  distCutoff,
  normalize,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type ScorerOptions,
  type Sequence,
  prepareMetric,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MaybeSequence,
  isNone,
  asSequence,
  type NormalizedScorer,
  type Scorer,
} from '../shared/scorerSupport.js'

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function preparedPrefixDistance(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return maximum(s1, s2) - commonPrefix(s1, s2)
}

/**
 * Elements outside the common prefix: `max(|s1|, |s2|)` minus the length of the
 * longest common prefix.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
function prefixDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  return distCutoff(maximum(a, b) - commonPrefix(a, b), options.scoreCutoff)
}

/**
 * Length of the longest common prefix of `s1` and `s2`.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function prefixSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  return simCutoff(commonPrefix(a, b), options.scoreCutoff)
}

/**
 * {@link prefixDistance} normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function prefixNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  return normDistCutoff(normalize(max - commonPrefix(a, b), max), options.scoreCutoff)
}

/**
 * {@link prefixSimilarity} normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function prefixNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  return normSimCutoff(1 - normalize(max - commonPrefix(a, b), max), options.scoreCutoff)
}

// Scorer flags let `process` tell distances from similarities.
export const prefixDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  prefixDistance_impl,
  DISTANCE_FLAGS,
  prepareMetric('distance', preparedPrefixDistance, maximum),
)
export const prefixSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  prefixSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareMetric('similarity', preparedPrefixDistance, maximum),
)
export const prefixNormalizedDistance: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    prefixNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareMetric('normalizedDistance', preparedPrefixDistance, maximum),
  )
export const prefixNormalizedSimilarity: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    prefixNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareMetric('normalizedSimilarity', preparedPrefixDistance, maximum),
  )
