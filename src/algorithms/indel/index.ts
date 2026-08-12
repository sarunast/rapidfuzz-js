import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  indelDistance,
  indelEditops,
  indelNormalizedDistance,
  indelNormalizedSimilarity,
  indelOpcodes,
  indelSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'indel.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'indel.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<'indel.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: indelNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
export const normalizedSimilarity: BuiltInMetric<
  'indel.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: indelNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
export { indelEditops as editops, indelOpcodes as opcodes }
