import type { Metric } from '../../core/metric.js'
import type { Direction, SimilarityConfiguration } from '../../core/types.js'
import { builtInMetric, type BuiltInMetric } from '../shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../shared/scorerSupport.js'
import { diceDistance, diceSimilarity, type DiceOptions } from './implementation.js'

export interface DiceDistanceConfiguration {
  readonly gramSize?: number | undefined
}
export interface DiceSimilarityConfiguration
  extends DiceDistanceConfiguration, SimilarityConfiguration {}

const GRAM_SIZE: readonly string[] = ['gramSize']

function diceMetric<D extends Direction, Config extends object, Brand>(
  implementation: MaybeSequenceMetricImplementation<DiceOptions>,
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
  'dice.distance',
  'distance',
  DiceDistanceConfiguration
> = /* @__PURE__ */ diceMetric(diceDistance, 'distance')
export const similarity: BuiltInMetric<
  'dice.similarity',
  'similarity',
  DiceDistanceConfiguration
> = /* @__PURE__ */ diceMetric(diceSimilarity, 'similarity')

// Dice is normalized by construction, so these are the same metrics under the
// names the other algorithms use. `typeof` carries the identity across instead
// of restating it, which is what keeps their prepared choices interchangeable.
export const normalizedDistance: typeof distance = distance
export const normalizedSimilarity: typeof similarity = similarity
