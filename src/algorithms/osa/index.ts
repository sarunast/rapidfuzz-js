import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  osaDistance,
  osaNormalizedDistance,
  osaNormalizedSimilarity,
  osaSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'osa.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'osa.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<'osa.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: osaNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
export const normalizedSimilarity: BuiltInMetric<
  'osa.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: osaNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
