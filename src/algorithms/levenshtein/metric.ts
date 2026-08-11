import {
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  withPreparedFlags,
  type ConfigurationCanonicalizer,
  type ConfigurationFlagsResolver,
  type NormalizedScorer,
  type Scorer,
  type ScorerFlags,
} from '../shared/scorerSupport.js'
import {
  levenshteinCosts,
  levenshteinDistanceImpl,
  levenshteinNormalizedDistanceImpl,
  levenshteinNormalizedSimilarityImpl,
  levenshteinSimilarityImpl,
} from './internal/engine.js'
import { prepareLevenshtein } from './prepare.js'
import type { LevenshteinOptions } from './types.js'

export type {
  LevenshteinCosts,
  LevenshteinOptions,
  LevenshteinWeights,
  LevenshteinWeightsInput,
} from './types.js'
export { levenshteinCosts } from './internal/engine.js'

/**
 * Weighted Levenshtein is symmetric only when insertion and deletion cost the
 * same, because swapping the arguments swaps those two operations.
 *
 * Scorer compilation calls this when weights are retained, which is the only
 * way they can reach a matrix. Reporting it through the flags is what lets
 * `scoreMatrix` decide whether it may mirror the lower triangle without knowing
 * that the option in question is spelled `weights`.
 */
function levenshteinConfigurationFlagsResolver(
  base: ScorerFlags,
): ConfigurationFlagsResolver {
  return (options) => {
    const { insertion, deletion } = levenshteinCosts(Reflect.get(options, 'weights'))
    return insertion === deletion ? base : { ...base, symmetric: false }
  }
}

/**
 * Snapshot baked `weights` so the caller can no longer reach them.
 *
 * Compilation records the symmetry of a weighting once, when it is retained.
 * Both spellings of the option are mutable — an array and an object
 * of three numbers — so without this a caller could bake a symmetric weighting,
 * mutate it afterwards, and leave a scorer that scores asymmetrically while its
 * recorded flags still permit `scoreMatrix` to mirror half the matrix. That is
 * a wrong number, not a stale one.
 *
 * {@link levenshteinCosts} already allocates, so the copy is what it returns;
 * freezing it costs one call, once per `createScorer`.
 */
const levenshteinConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  const weights = Reflect.get(options, 'weights')
  if (weights == null) return options
  return { ...options, weights: Object.freeze(levenshteinCosts(weights)) }
}

// Scorer flags let generic batch and search code specialize by direction.
export const levenshteinDistance: Scorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinDistanceImpl,
    DISTANCE_FLAGS,
    prepareLevenshtein('distance'),
    {
      configurationFlags:
        /* @__PURE__ */ levenshteinConfigurationFlagsResolver(DISTANCE_FLAGS),
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinSimilarity: Scorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinSimilarityImpl,
    SIMILARITY_FLAGS,
    prepareLevenshtein('similarity'),
    {
      configurationFlags:
        /* @__PURE__ */ levenshteinConfigurationFlagsResolver(SIMILARITY_FLAGS),
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinNormalizedDistance: NormalizedScorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedDistanceImpl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareLevenshtein('normalizedDistance'),
    {
      configurationFlags: /* @__PURE__ */ levenshteinConfigurationFlagsResolver(
        NORMALIZED_DISTANCE_FLAGS,
      ),
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinNormalizedSimilarity: NormalizedScorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedSimilarityImpl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareLevenshtein('normalizedSimilarity'),
    {
      configurationFlags: /* @__PURE__ */ levenshteinConfigurationFlagsResolver(
        NORMALIZED_SIMILARITY_FLAGS,
      ),
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
