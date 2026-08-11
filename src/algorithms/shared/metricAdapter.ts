import { type Metric } from '../../core/metric.js'
import { COMPILE, type MetricCompilation } from '../../core/protocol.js'
import { validatePair, validateSequence } from '../../core/sequence.js'
import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Sequence,
} from '../../core/types.js'
import {
  configurationSymmetryOf,
  configurationCanonicalizerOf,
  PREPARE_SCORER,
  type ErasedMetricImplementation,
  type PreparedCapability,
  type ScorerOptions,
} from './scorerSupport.js'

interface BuiltInMetricOptions<D extends Direction> {
  readonly implementation: ErasedMetricImplementation & PreparedCapability
  readonly directImplementation?:
    | ((a: MaybeSequence, b: MaybeSequence) => number)
    | undefined
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly configurationKeys?: readonly string[] | undefined
}

// Reads `unknown` because `createScorer(levenshtein.distance, null)` reaches
// here from JavaScript, where the hook's `Config | undefined` proves nothing.
// `Object.keys` answers `[]` for a number and a boolean alike, so a primitive
// would otherwise pass for "no configuration" — and a string reached
// `Reflect.get` and failed with an error about our own internals.
function configurationObject(given: unknown): object {
  if (given === undefined) return {}
  if (typeof given !== 'object' || given === null) {
    throw new TypeError('metric configuration must be an object')
  }
  return given
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
  options: BuiltInMetricOptions<D>,
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
    const { record: initial, missing } = configurationRecord(
      configurationObject(given),
      options.direction,
      options.configurationKeys ?? [],
    )
    const canonicalizer = configurationCanonicalizerOf(options.implementation)
    const record = canonicalizer === null ? initial : canonicalizer(initial)
    const configured = Object.keys(record).length !== 0
    const symmetry = configurationSymmetryOf(options.implementation)
    const symmetric = symmetry?.(record) ?? true
    const preparation = options.implementation[PREPARE_SCORER](record)
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
      prepareQuery: preparation.prepareQuery,
      prepareChoice: preparation.prepareChoice,
    }
  }
  return Object.assign(direct, {
    [COMPILE]: compile,
  })
}
