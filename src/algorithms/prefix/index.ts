import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  prefixDistance,
  prefixNormalizedDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'prefix.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixDistance,
    directImplementation: prefixDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'prefix.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixSimilarity,
    directImplementation: prefixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<'prefix.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: prefixNormalizedDistance,
    directImplementation: prefixNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
export const normalizedSimilarity: BuiltInMetric<
  'prefix.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: prefixNormalizedSimilarity,
  directImplementation: prefixNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
