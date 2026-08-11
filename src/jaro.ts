import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import { jaroSimilarity } from './distance/jaro.js'

export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: jaroSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
