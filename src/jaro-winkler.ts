import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import { jaroWinklerSimilarity } from './distance/jaroWinkler.js'

export interface JaroWinklerConfiguration extends SimilarityConfiguration {
  readonly prefixWeight?: number | undefined
}

export const similarity: Metric<'similarity', JaroWinklerConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: jaroWinklerSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
