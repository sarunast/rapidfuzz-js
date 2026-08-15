import type { MaybeSequenceMetricImplementation } from '../core/scoring/builtIn/implementation.js'
import { builtInMetric } from '../core/scoring/builtIn/metric.js'
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
