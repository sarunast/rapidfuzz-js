import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedSimilarity,
} from './distance/damerauLevenshtein.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: damerauLevenshteinDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: damerauLevenshteinNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
