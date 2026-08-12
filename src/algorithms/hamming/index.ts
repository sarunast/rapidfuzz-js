import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import {
  hammingDistance,
  hammingEditops,
  hammingNormalizedDistance,
  hammingNormalizedSimilarity,
  hammingOpcodes,
  hammingSimilarity,
  type HammingOptions,
} from './implementation.js'

export interface HammingDistanceConfiguration {
  readonly pad?: boolean | undefined
}
export interface HammingSimilarityConfiguration
  extends HammingDistanceConfiguration, SimilarityConfiguration {}

const PAD: readonly string[] = ['pad']

function hammingMetric<D extends Direction, Config extends object, Brand>(
  implementation: MaybeSequenceMetricImplementation<HammingOptions>,
  direction: D,
  bounds: readonly [number, number],
): Metric<D, Config, Brand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds,
    configurationKeys: PAD,
  })
}

export const distance: BuiltInMetric<
  'hamming.distance',
  'distance',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingDistance, 'distance', [
  0,
  Number.POSITIVE_INFINITY,
])
export const similarity: BuiltInMetric<
  'hamming.similarity',
  'similarity',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingSimilarity, 'similarity', [
  0,
  Number.POSITIVE_INFINITY,
])
export const normalizedDistance: BuiltInMetric<
  'hamming.normalizedDistance',
  'distance',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingNormalizedDistance, 'distance', [0, 1])
export const normalizedSimilarity: BuiltInMetric<
  'hamming.normalizedSimilarity',
  'similarity',
  HammingDistanceConfiguration
> = /* @__PURE__ */ hammingMetric(hammingNormalizedSimilarity, 'similarity', [0, 1])
export { hammingEditops as editops, hammingOpcodes as opcodes }
