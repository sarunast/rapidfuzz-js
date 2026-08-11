import type { Metric } from '../../core/metric.js'
import type { SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric } from '../shared/metricAdapter.js'
import {
  hammingDistance,
  hammingEditops,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingOpcodes,
  hammingSimilarity,
} from './implementation.js'

export interface HammingDistanceConfiguration {
  readonly pad?: boolean | undefined
}
export interface HammingSimilarityConfiguration
  extends HammingDistanceConfiguration, SimilarityConfiguration {}

export const distance: Metric<'distance', HammingDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingDistance,
    directImplementation: hammingDistance,
    direction: 'distance',
    bounds: [0, Number.POSITIVE_INFINITY],
    configurationKeys: ['pad'],
  })
export const similarity: Metric<'similarity', HammingSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingSimilarity,
    directImplementation: hammingSimilarity,
    direction: 'similarity',
    bounds: [0, Number.POSITIVE_INFINITY],
    configurationKeys: ['pad'],
  })
export const normalizedDistance: Metric<'distance', HammingDistanceConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingNormalizedDistance,
    directImplementation: hammingNormalizedDistance,
    direction: 'distance',
    bounds: [0, 1],
    configurationKeys: ['pad'],
  })
export const normalizedSimilarity: Metric<'similarity', HammingSimilarityConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: hammingNormalizedSimilarity,
    directImplementation: hammingNormalizedSimilarity,
    direction: 'similarity',
    bounds: [0, 1],
    configurationKeys: ['pad'],
  })
export { hammingEditops as editops, hammingOpcodes as opcodes }
