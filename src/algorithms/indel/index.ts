import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  indelDistance,
  indelEditops,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelOpcodes,
  indelSimilarity,
} from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: indelDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: indelNormalizedDistance,
  direction: 'distance',
  bounds: [0, 1],
})
export const normalizedSimilarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { indelEditops as editops, indelOpcodes as opcodes }
