import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import {
  lcsSeqDistance,
  lcsSeqEditops,
  lcsSeqNormalizedSimilarity,
  lcsSeqOpcodes,
} from './distance/lcsSeq.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: lcsSeqDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: lcsSeqNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { lcsSeqEditops as editops, lcsSeqOpcodes as opcodes }
