import { COMPILE, METRIC_CONFIGURATION, type MetricCompilation } from './protocol.js'
import type { Direction, MaybeSequence } from './types.js'

export interface Metric<
  D extends Direction,
  Config extends object = Record<never, never>,
> {
  (a: MaybeSequence, b: MaybeSequence): number
  readonly [METRIC_CONFIGURATION]: (configuration: Config) => Config
  readonly [COMPILE]: (configuration: Config | undefined) => MetricCompilation<D>
}

export function isBuiltInMetric(value: object): value is Metric<Direction, object> {
  return COMPILE in value && typeof value[COMPILE] === 'function'
}
