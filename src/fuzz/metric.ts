import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/scoring/metric.js'
import type { FuzzOptions } from './types.js'

const BOUNDS: readonly [number, number] = [0, 100]

export function fuzzMetric<TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<FuzzOptions>,
): Metric<'similarity', TConfig, TBrand> {
  return builtInMetric({
    implementation,
    direction: 'similarity',
    bounds: BOUNDS,
  })
}
