import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import { postfixDistance, postfixNormalizedSimilarity } from './distance/postfix.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: postfixDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: postfixNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
