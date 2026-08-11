import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import {
  levenshteinDistance,
  levenshteinEditops,
  levenshteinNormalizedSimilarity,
  levenshteinOpcodes,
  type LevenshteinWeightsInput,
} from './distance/levenshtein.js'

export type { LevenshteinCosts, LevenshteinWeights } from './distance/levenshtein.js'

export interface LevenshteinDistanceConfiguration {
  readonly weights?: LevenshteinWeightsInput | undefined
}

export interface LevenshteinSimilarityConfiguration
  extends LevenshteinDistanceConfiguration,
    SimilarityConfiguration {}

export const distance: Metric<'distance', LevenshteinDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: levenshteinDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })

export const similarity: Metric<'similarity', LevenshteinSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: levenshteinNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })

export { levenshteinEditops as editops, levenshteinOpcodes as opcodes }
