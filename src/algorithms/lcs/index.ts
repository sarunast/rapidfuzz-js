import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import {
  lcsSeqDistance,
  lcsSeqEditops,
  lcsSeqNormalizedDistance,
  lcsSeqNormalizedSimilarity,
  lcsSeqOpcodes,
  lcsSeqSimilarity,
} from './implementation.js'

export const distance: BuiltInMetric<'lcs.distance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: BuiltInMetric<'lcs.similarity', 'similarity'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const normalizedDistance: BuiltInMetric<'lcs.normalizedDistance', 'distance'> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
  })
export const normalizedSimilarity: BuiltInMetric<
  'lcs.normalizedSimilarity',
  'similarity'
> = /* @__PURE__ */ builtInMetric({
  implementation: lcsSeqNormalizedSimilarity,
  direction: 'similarity',
  bounds: [0, 1],
})
export { lcsSeqEditops as editops, lcsSeqOpcodes as opcodes }
