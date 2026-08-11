import { COMPILE, type MetricCompilation } from './protocol.js'
import type { Direction, MaybeSequence } from './types.js'

export interface Metric<
  D extends Direction,
  Config extends object = Record<never, never>,
> {
  (a: MaybeSequence, b: MaybeSequence): number
  readonly [COMPILE]: (configuration: Config | undefined) => MetricCompilation<D>
}

export function isBuiltInMetric<D extends Direction, Config extends object>(
  value: Metric<D, Config> | ((a: MaybeSequence, b: MaybeSequence) => number),
): value is Metric<D, Config> {
  return COMPILE in value && typeof value[COMPILE] === 'function'
}
