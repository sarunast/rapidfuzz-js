import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MetricImplementation } from '../shared/scorerSupport.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
  type LevenshteinWeightsInput,
} from './metric.js'
import type { LevenshteinOptions } from './types.js'

export interface LevenshteinDistanceConfiguration {
  readonly weights?: LevenshteinWeightsInput | undefined
}

export interface LevenshteinSimilarityConfiguration
  extends LevenshteinDistanceConfiguration, SimilarityConfiguration {}

const WEIGHTS: readonly string[] = ['weights']

function levenshteinMetric<D extends Direction, Config extends object, Brand>(
  implementation: MetricImplementation<LevenshteinOptions>,
  direction: D,
  bounds: readonly [number, number],
): Metric<D, Config, Brand> {
  return builtInMetric({
    implementation,
    direction,
    bounds,
    configurationKeys: WEIGHTS,
  })
}

export const distance: BuiltInMetric<
  'levenshtein.distance',
  'distance',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinDistance, 'distance', [
  0,
  Number.POSITIVE_INFINITY,
])
export const similarity: BuiltInMetric<
  'levenshtein.similarity',
  'similarity',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinSimilarity, 'similarity', [
  0,
  Number.POSITIVE_INFINITY,
])
export const normalizedDistance: BuiltInMetric<
  'levenshtein.normalizedDistance',
  'distance',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(levenshteinNormalizedDistance, 'distance', [0, 1])
export const normalizedSimilarity: BuiltInMetric<
  'levenshtein.normalizedSimilarity',
  'similarity',
  LevenshteinDistanceConfiguration
> = /* @__PURE__ */ levenshteinMetric(
  levenshteinNormalizedSimilarity,
  'similarity',
  [0, 1],
)
