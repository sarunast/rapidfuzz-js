import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  hammingDistance,
  hammingEditops,
  hammingNormalizedSimilarity,
  hammingOpcodes,
} from './implementation.js'

export interface HammingDistanceConfiguration {
  readonly pad?: boolean | undefined
}
export interface HammingSimilarityConfiguration
  extends HammingDistanceConfiguration, SimilarityConfiguration {}

export const distance: Metric<'distance', HammingDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
  })
export const similarity: Metric<'similarity', HammingSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
  })
export { hammingEditops as editops, hammingOpcodes as opcodes }
