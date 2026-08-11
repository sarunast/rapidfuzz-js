import { type Metric } from '../../core/metric.js'
import {
  COMPILE,
  type MetricCompilation,
  type PreparedKernel,
} from '../../core/protocol.js'
import { validatePair, validateSequence } from '../../core/sequence.js'
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
  type ScorerOptions,
} from './scorerSupport.js'

interface BuiltInMetricOptions<D extends Direction, Config extends object> {
  readonly implementation: PreparedErasedScorer
  readonly directImplementation?:
    | ((a: MaybeSequence, b: MaybeSequence) => number)
    | undefined
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly configurationKeys?: readonly string[] | undefined
  readonly canonicalize?: ((configuration: Config) => Config) | undefined
}

function configurationRecord<D extends Direction>(
  configuration: object,
  direction: D,
  configurationKeys: readonly string[],
): {
  readonly record: Readonly<Record<string, unknown>> & ScorerOptions
  readonly missing: MissingPolicy
} {
  const record: Record<string, unknown> & ScorerOptions = {}
  let missing: MissingPolicy = 'compatible'
  for (const key of Object.keys(configuration)) {
    const value = Reflect.get(configuration, key)
    if (key === 'missing') {
      if (direction !== 'similarity') {
        throw new TypeError("unknown metric configuration key 'missing'")
      }
      if (value === 'compatible') missing = 'compatible'
      else if (value === 'throw') missing = 'throw'
      else throw new TypeError("missing must be 'compatible' or 'throw'")
    } else {
      if (!configurationKeys.includes(key)) {
        throw new TypeError(`unknown metric configuration key '${key}'`)
      }
      record[key] = value
    }
  }
  return { record, missing }
}

export function builtInMetric<D extends Direction, Config extends object>(
  options: BuiltInMetricOptions<D, Config>,
): Metric<D, Config> {
  // Keep this as a normal direct call: `Reflect.apply` measured 5-7% slower
  // over short-string comparisons.
  const implementation = options.implementation
  const direct =
    options.directImplementation ??
    ((a: MaybeSequence, b: MaybeSequence): number => {
      if (a == null || b == null) {
        if (options.direction === 'similarity') return 0
        throw new TypeError('missing sequences are not supported by this scorer')
      }
      // Keep the direct Metric path allocation-free. `validatePair` returns a
      // tuple for generic callers; constructing that tuple for every short-string
      // comparison was more expensive than the validation itself.
      if (typeof a !== 'string') validateSequence(a)
      if (typeof b !== 'string') validateSequence(b)
      return implementation(a, b)
    })
  const compile = (given: Config | undefined): MetricCompilation<D> => {
    const canonical: object =
      given === undefined ? {} : (options.canonicalize?.(given) ?? given)
    const { record: initial, missing } = configurationRecord(
      canonical,
      options.direction,
      options.configurationKeys ?? [],
    )
    const canonicalizer = configurationCanonicalizerOf(options.implementation)
    const record = canonicalizer === null ? initial : canonicalizer(initial)
    const configured = Object.keys(record).length !== 0
    const flags = configurationFlagsOf(options.implementation)
    const symmetric = flags?.(record).symmetric ?? true
    const prepare = options.implementation[PREPARE_SCORER]
    // Batch and driver loops call rawScore thousands of times with one fixed
    // threshold; a one-entry cache keeps the cutoff-bearing options from being
    // rebuilt per pair. The threshold changes between loops, not inside them.
    let cutoffOptions: (Readonly<Record<string, unknown>> & ScorerOptions) | null = null
    let cutoffThreshold = 0
    const rawScore = (a: Sequence, b: Sequence, threshold: number | null): number => {
      if (threshold === null) {
        return implementation(a, b, configured ? record : undefined)
      }
      if (cutoffOptions === null || cutoffThreshold !== threshold) {
        cutoffOptions = { ...record, scoreCutoff: threshold }
        cutoffThreshold = threshold
      }
      return implementation(a, b, cutoffOptions)
    }
    const score = (
      a: MaybeSequence,
      b: MaybeSequence,
      threshold: number | null,
    ): number => {
      const pair = validatePair(a, b, options.direction, missing)
      return pair === null ? 0 : rawScore(pair[0], pair[1], threshold)
    }
    const prepareQuery = (query: Sequence): PreparedKernel => prepare(query, record)
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
      rawScore,
      prepareQuery,
      prepareChoice: (choice) => choicePreparer(choice),
    }
  }
  return Object.assign(direct, {
    [COMPILE]: compile,
  })
}
