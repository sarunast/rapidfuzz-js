import { type Metric } from '../../core/metric.js'
import {
  COMPILE,
  type MetricCompilation,
  type PreparedKernel,
} from '../../core/protocol.js'
import { validatePair } from '../../core/sequence.js'
import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Sequence,
} from '../../core/types.js'
import {
  configurationFlagsOf,
  configurationCanonicalizerOf,
  PREPARE_CHOICE,
  PREPARE_SCORER,
  type PreparedErasedScorer,
} from './scorerSupport.js'

interface BuiltInMetricOptions<D extends Direction, Config extends object> {
  readonly implementation: PreparedErasedScorer
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly canonicalize?: ((configuration: Config) => Config) | undefined
}

function configurationRecord(configuration: object): {
  readonly record: Readonly<Record<string, unknown>>
  readonly missing: MissingPolicy
} {
  const record: Record<string, unknown> = {}
  let missing: MissingPolicy = 'compatible'
  for (const key of Object.keys(configuration)) {
    const value = Reflect.get(configuration, key)
    if (key === 'missing') {
      if (value === 'compatible') missing = 'compatible'
      else if (value === 'throw') missing = 'throw'
      else throw new TypeError("missing must be 'compatible' or 'throw'")
    } else record[key] = value
  }
  return { record, missing }
}

function callImplementation(
  implementation: PreparedErasedScorer,
  a: Sequence,
  b: Sequence,
  options: Readonly<Record<string, unknown>> | undefined,
): number {
  const args = options === undefined ? [a, b] : [a, b, options]
  return Reflect.apply(implementation, undefined, args)
}

export function builtInMetric<D extends Direction, Config extends object>(
  options: BuiltInMetricOptions<D, Config>,
): Metric<D, Config> {
  const direct = (a: MaybeSequence, b: MaybeSequence): number => {
    const pair = validatePair(a, b, options.direction, 'compatible')
    return pair === null
      ? 0
      : callImplementation(options.implementation, pair[0], pair[1], undefined)
  }
  const compile = (given: Config | undefined): MetricCompilation<D> => {
    const empty: Record<string, never> = {}
    const canonical =
      given === undefined ? empty : (options.canonicalize?.(given) ?? given)
    const { record: initial, missing } = configurationRecord(canonical)
    const canonicalizer = configurationCanonicalizerOf(options.implementation)
    const record = canonicalizer === null ? initial : canonicalizer(initial)
    const flags = configurationFlagsOf(options.implementation)
    const symmetric = flags?.(record).symmetric ?? true
    const prepare = options.implementation[PREPARE_SCORER]
    const score = (
      a: MaybeSequence,
      b: MaybeSequence,
      threshold: number | null,
    ): number => {
      const pair = validatePair(a, b, options.direction, missing)
      if (pair === null) return 0
      const callOptions =
        threshold === null ? record : { ...record, scoreCutoff: threshold }
      return callImplementation(options.implementation, pair[0], pair[1], callOptions)
    }
    const prepareQuery = (query: Sequence): PreparedKernel => {
      const prepared = prepare(query, record)
      return (choice, threshold) => prepared(choice, threshold)
    }
    const choicePreparer = prepare[PREPARE_CHOICE]
    return {
      direction: options.direction,
      bounds: options.bounds,
      symmetric,
      trusted: true,
      validate: (a, b) => {
        validatePair(a, b, options.direction, missing)
      },
      score,
      prepareQuery,
      prepareChoice: (choice) => choicePreparer(choice),
    }
  }
  return Object.assign(direct, {
    [COMPILE]: compile,
  })
}
