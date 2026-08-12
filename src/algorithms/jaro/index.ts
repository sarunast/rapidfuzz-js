import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import { jaroDistance, jaroSimilarity } from './implementation.js'

export const distance: BuiltInMetric<'jaro.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroDistance,
    directImplementation: jaroDistance,
    direction: 'distance',
    bounds: [0, 1],
  })

export const similarity: BuiltInMetric<'jaro.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: jaroSimilarity,
    directImplementation: jaroSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })

// Jaro is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
export const normalizedDistance: typeof distance = distance
export const normalizedSimilarity: typeof similarity = similarity
