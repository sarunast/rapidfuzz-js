import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: postfixDistance,
  directImplementation: postfixDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixSimilarity,
    directImplementation: postfixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: postfixNormalizedDistance,
  directImplementation: postfixNormalizedDistance,
  direction: 'distance',
  bounds: [0, 1],
})
export const normalizedSimilarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixNormalizedSimilarity,
    directImplementation: postfixNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
