import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import { jaroWinklerDistance, jaroWinklerSimilarity } from './implementation.js'

export interface JaroWinklerConfiguration extends SimilarityConfiguration {
  readonly prefixWeight?: number | undefined
}

export interface JaroWinklerDistanceConfiguration {
  readonly prefixWeight?: number | undefined
}

export const distance: Metric<'distance', JaroWinklerDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroWinklerDistance,
    directImplementation: jaroWinklerDistance,
    direction: 'distance',
    bounds: [0, 1],
    configurationKeys: ['prefixWeight'],
  })

export const similarity: Metric<'similarity', JaroWinklerConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroWinklerSimilarity,
    directImplementation: jaroWinklerSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
    configurationKeys: ['prefixWeight'],
  })

export const normalizedDistance: Metric<'distance', JaroWinklerDistanceConfiguration> =
  distance
export const normalizedSimilarity: Metric<'similarity', JaroWinklerConfiguration> =
  similarity
