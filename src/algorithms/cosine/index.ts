import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { cosineDistance, cosineSimilarity, type CosineOptions } from './implementation.js'

export interface CosineDistanceConfiguration {
  readonly gramSize?: number | undefined
}
export interface CosineSimilarityConfiguration
  extends CosineDistanceConfiguration, SimilarityConfiguration {}

const GRAM_SIZE: readonly string[] = ['gramSize']

function cosineMetric<D extends Direction, Config extends object, Brand>(
  implementation: MaybeSequenceMetricImplementation<CosineOptions>,
  direction: D,
): Metric<D, Config, Brand> {
  return builtInMetric({
    implementation,
    directImplementation: implementation,
    direction,
    bounds: [0, 1],
    configurationKeys: GRAM_SIZE,
  })
}

export const distance: BuiltInMetric<
  'cosine.distance',
  'distance',
  CosineDistanceConfiguration
> = /* @__PURE__ */ cosineMetric(cosineDistance, 'distance')
export const similarity: BuiltInMetric<
  'cosine.similarity',
  'similarity',
  CosineDistanceConfiguration
> = /* @__PURE__ */ cosineMetric(cosineSimilarity, 'similarity')

// Cosine is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
export const normalizedDistance: typeof distance = distance
export const normalizedSimilarity: typeof similarity = similarity
