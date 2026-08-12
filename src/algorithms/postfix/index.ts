import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  postfixDistance,
  postfixNormalizedDistance,
  postfixNormalizedSimilarity,
  postfixSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'postfix.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixDistance,
    directImplementation: postfixDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'postfix.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixSimilarity,
    directImplementation: postfixSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<'postfix.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: postfixNormalizedDistance,
    directImplementation: postfixNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
export const normalizedSimilarity: BuiltInMetric<
  'postfix.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: postfixNormalizedSimilarity,
  directImplementation: postfixNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
