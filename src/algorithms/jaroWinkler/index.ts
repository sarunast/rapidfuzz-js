import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { jaroWinklerDistance, jaroWinklerSimilarity } from './implementation.js'

export interface JaroWinklerConfiguration extends SimilarityConfiguration {
  readonly prefixWeight?: number | undefined
}

export interface JaroWinklerDistanceConfiguration {
  readonly prefixWeight?: number | undefined
}

const PREFIX_WEIGHT: readonly string[] = ['prefixWeight']

function jaroWinklerMetric<D extends Direction, Config extends object, Brand>(
  implementation: MaybeSequenceMetricImplementation,
  direction: D,
): Metric<D, Config, Brand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: PREFIX_WEIGHT,
  })
}

export const distance: BuiltInMetric<
  'jaroWinkler.distance',
  'distance',
  JaroWinklerDistanceConfiguration
> = /* @__PURE__ */ jaroWinklerMetric(jaroWinklerDistance, 'distance')
export const similarity: BuiltInMetric<
  'jaroWinkler.similarity',
  'similarity',
  JaroWinklerDistanceConfiguration
> = /* @__PURE__ */ jaroWinklerMetric(jaroWinklerSimilarity, 'similarity')

// Jaro-Winkler is normalized by construction, so these are the same metrics
// under the names the other algorithms use. `typeof` carries the identity
// across instead of restating it, which is what keeps their prepared choices
// interchangeable.
export const normalizedDistance: typeof distance = distance
export const normalizedSimilarity: typeof similarity = similarity
