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

export function builtInMetric<D extends Direction, Config extends object>(
  options: BuiltInMetricOptions<D, Config>,
): Metric<D, Config> {
  // `builtInMetric` is the package-owned registration boundary: every caller
  // supplies an implementation whose first two parameters are Sequences. Bind
  // that fact once so the cheapest public Metric path remains a normal direct
  // call; `Reflect.apply` measured 5-7% slower over short-string comparisons.
  // oxlint-disable-next-line typescript/consistent-type-assertions -- registration proves this private callable shape once
  const implementation = options.implementation as unknown as (
    a: Sequence,
    b: Sequence,
    options?: Readonly<Record<string, unknown>>,
  ) => number
  const direct = (a: MaybeSequence, b: MaybeSequence): number => {
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
  }
  const compile = (given: Config | undefined): MetricCompilation<D> => {
    const empty: Record<string, never> = {}
    const canonical =
      given === undefined ? empty : (options.canonicalize?.(given) ?? given)
    const { record: initial, missing } = configurationRecord(canonical)
    const canonicalizer = configurationCanonicalizerOf(options.implementation)
    const record = canonicalizer === null ? initial : canonicalizer(initial)
    const configured = Object.keys(record).length !== 0
    const flags = configurationFlagsOf(options.implementation)
    const symmetric = flags?.(record).symmetric ?? true
    const prepare = options.implementation[PREPARE_SCORER]
    const rawScore = (a: Sequence, b: Sequence, threshold: number | null): number => {
      const callOptions =
        threshold === null
          ? configured
            ? record
            : undefined
          : { ...record, scoreCutoff: threshold }
      return implementation(a, b, callOptions)
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
