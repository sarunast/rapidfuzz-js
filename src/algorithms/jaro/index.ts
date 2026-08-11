import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import { jaroSimilarity } from './implementation.js'

export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
