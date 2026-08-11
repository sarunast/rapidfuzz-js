import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
import {
  hammingDistance,
  hammingEditops,
  hammingNormalizedSimilarity,
  hammingOpcodes,
} from './distance/hamming.js'

export interface HammingDistanceConfiguration {
  readonly pad?: boolean | undefined
}
export interface HammingSimilarityConfiguration
  extends HammingDistanceConfiguration,
    SimilarityConfiguration {}

export const distance: Metric<'distance', HammingDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: hammingDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: Metric<'similarity', HammingSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: hammingNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { hammingEditops as editops, hammingOpcodes as opcodes }
