import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import { osaDistance, osaNormalizedSimilarity } from './distance/osa.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: osaDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: osaNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
