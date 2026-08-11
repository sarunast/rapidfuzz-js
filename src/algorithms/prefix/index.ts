import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: prefixDistance,
  directImplementation: prefixDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixSimilarity,
    directImplementation: prefixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: prefixNormalizedDistance,
  directImplementation: prefixNormalizedDistance,
  direction: 'distance',
  bounds: [0, 1],
})
export const normalizedSimilarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixNormalizedSimilarity,
    directImplementation: prefixNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
