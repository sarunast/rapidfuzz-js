import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  lcsSeqDistance,
  lcsSeqEditops,
  lcsSeqNormalizedSimilarity,
  lcsSeqOpcodes,
} from './implementation.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  implementation: lcsSeqDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: lcsSeqNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { lcsSeqEditops as editops, lcsSeqOpcodes as opcodes }
