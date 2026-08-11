import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import {
  indelDistance,
  indelEditops,
  indelNormalizedSimilarity,
  indelOpcodes,
} from './distance/indel.js'

export const distance: Metric<'distance'> = /* @__PURE__ */ builtInMetric({
  legacy: indelDistance,
  direction: 'distance',
  bounds: [0, Number.POSITIVE_INFINITY],
})
export const similarity: Metric<'similarity', SimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: indelNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { indelEditops as editops, indelOpcodes as opcodes }
