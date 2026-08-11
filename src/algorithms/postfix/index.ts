import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import { postfixDistance, postfixNormalizedSimilarity } from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: postfixDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
