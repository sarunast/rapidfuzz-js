import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  levenshteinDistance,
  levenshteinNormalizedSimilarity,
  type LevenshteinWeightsInput,
} from './metric.js'

export interface LevenshteinDistanceConfiguration {
  readonly weights?: LevenshteinWeightsInput | undefined
}

export interface LevenshteinSimilarityConfiguration
  extends LevenshteinDistanceConfiguration, SimilarityConfiguration {}

export const distance: Metric<'distance', LevenshteinDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: levenshteinDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
    configurationKeys: ['weights'],
  })

export const similarity: Metric<'similarity', LevenshteinSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: levenshteinNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
    configurationKeys: ['weights'],
  })
