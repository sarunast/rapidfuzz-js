import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  damerauLevenshteinDistance,
  damerauLevenshteinNormalizedDistance,
  damerauLevenshteinNormalizedSimilarity,
  damerauLevenshteinSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'damerauLevenshtein.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: damerauLevenshteinDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'damerauLevenshtein.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: damerauLevenshteinSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<
  'damerauLevenshtein.normalizedDistance',
  'distance'
> = /* @__PURE__ */ builtInMetric({
  implementation: damerauLevenshteinNormalizedDistance,
  direction: 'distance',
  bounds: [0, 1],
})
export const normalizedSimilarity: BuiltInMetric<
  'damerauLevenshtein.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: damerauLevenshteinNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
