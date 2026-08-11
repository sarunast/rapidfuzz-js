import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import { jaroWinklerSimilarity } from './implementation.js'

export interface JaroWinklerConfiguration extends SimilarityConfiguration {
  readonly prefixWeight?: number | undefined
}

export const similarity: Metric<'similarity', JaroWinklerConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroWinklerSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
    configurationKeys: ['prefixWeight'],
  })
