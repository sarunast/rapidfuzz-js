import { builtInMetric } from '../../algorithms/shared/metricAdapter.js'
import type { MaybeSequenceMetricImplementation } from '../../algorithms/shared/scorerSupport.js'
import type { Metric } from '../../core/scoring/metric.js'
import type { FuzzOptions } from '../types.js'

// Every fuzz scorer reports a percentage.
const BOUNDS: readonly [number, number] = [0, 100]

/**
 * The two facts every fuzz metric shares: the direction, and the percentage
 * scale. Which metric it is stays with the declaration, in its type.
 */
export function fuzzMetric<TConfig extends object, TBrand>(
  implementation: MaybeSequenceMetricImplementation<FuzzOptions>,
): Metric<'similarity', TConfig, TBrand> {
  return builtInMetric({
    implementation,
    direction: 'similarity',
    bounds: BOUNDS,
  })
}
