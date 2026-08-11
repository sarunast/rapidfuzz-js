import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import { prefixDistance, prefixNormalizedSimilarity } from './distance/prefix.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: prefixDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: prefixNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
