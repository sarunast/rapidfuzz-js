import {
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  withPreparedFlags,
  type ConfigurationCanonicalizer,
  type ConfigurationSymmetryResolver,
  type MetricImplementation,
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

const levenshteinConfigurationSymmetry: ConfigurationSymmetryResolver = (options) => {
  const { insertion, deletion } = levenshteinCosts(Reflect.get(options, 'weights'))
  return insertion === deletion
}

const levenshteinConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  const weights = Reflect.get(options, 'weights')
  if (weights == null) return options
  return { ...options, weights: Object.freeze(levenshteinCosts(weights)) }
}

export const levenshteinDistance: MetricImplementation<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinDistanceImpl,
    DISTANCE_FLAGS,
    prepareLevenshtein('distance'),
    {
      configurationSymmetry: levenshteinConfigurationSymmetry,
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinSimilarity: MetricImplementation<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinSimilarityImpl,
    SIMILARITY_FLAGS,
    prepareLevenshtein('similarity'),
    {
      configurationSymmetry: levenshteinConfigurationSymmetry,
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinNormalizedDistance: MetricImplementation<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedDistanceImpl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareLevenshtein('normalizedDistance'),
    {
      configurationSymmetry: levenshteinConfigurationSymmetry,
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
export const levenshteinNormalizedSimilarity: MetricImplementation<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedSimilarityImpl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareLevenshtein('normalizedSimilarity'),
    {
      configurationSymmetry: levenshteinConfigurationSymmetry,
      configurationCanonicalizer: levenshteinConfigurationCanonicalizer,
    },
  )
