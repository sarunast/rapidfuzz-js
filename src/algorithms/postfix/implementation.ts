import {
  commonSuffix,
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
  const [a, b] = conv(s1, s2, options.processor)
  return distCutoff(maximum(a, b) - commonSuffix(a, b), options.scoreCutoff)
}

/**
 * Length of the longest common suffix of `s1` and `s2`.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function postfixSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  return simCutoff(commonSuffix(a, b), options.scoreCutoff)
}

/**
 * {@link postfixDistance} normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function postfixNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  return normDistCutoff(normalize(max - commonSuffix(a, b), max), options.scoreCutoff)
}

/**
 * {@link postfixSimilarity} normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function postfixNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  return normSimCutoff(1 - normalize(max - commonSuffix(a, b), max), options.scoreCutoff)
}

// Scorer flags let `process` tell distances from similarities.
export const postfixDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  postfixDistance_impl,
  DISTANCE_FLAGS,
  prepareMetric('distance', preparedPostfixDistance, maximum),
)
export const postfixSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
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
