import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import { jaroDistance, jaroSimilarity } from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: jaroDistance,
  directImplementation: jaroDistance,
  direction: 'distance',
  bounds: [0, 1],
})

export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroSimilarity,
    directImplementation: jaroSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })

export const normalizedDistance: Metric<'distance'> = distance
export const normalizedSimilarity: Metric<'similarity', SimilarityConfiguration> =
  similarity
