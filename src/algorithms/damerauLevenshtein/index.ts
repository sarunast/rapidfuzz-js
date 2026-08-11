import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedSimilarity,
} from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: damerauLevenshteinDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: damerauLevenshteinNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
