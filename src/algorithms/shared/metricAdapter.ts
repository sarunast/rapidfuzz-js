import { COMPILE, type MetricCompilation } from '../../core/scoring/compilation.js'
import type { Metric, NoConfiguration } from '../../core/scoring/metric.js'
import { snapshotSequence, validatePair, validateSequence } from '../../core/sequence.js'
import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Sequence,
  SimilarityConfiguration,
} from '../../core/types.js'
import {
  configurationSymmetryOf,
  configurationCanonicalizerOf,
  PREPARE_SCORER,
  type ErasedMetricImplementation,
  type PreparedCapability,
  type ScorerOptions,
} from './scorerSupport.js'

/**
 * The type of a metric this package built, named by `TId`.
 *
 * The name is the whole of the metric's identity, and it is the brand as-is:
 * the id literal is what makes a prepared choice belong to one metric and not
 * another, and a bare literal survives a consumer's declaration emit where a
 * wrapper type of ours could not be named. `TConfig` is what the metric itself
 * configures — the direction decides the rest, since `missing` is accepted by
 * a similarity and refused by a distance.
 */
export type BuiltInMetric<
  TId extends string,
  TDirection extends Direction,
  TConfig extends object = NoConfiguration,
> = Metric<
  TDirection,
  TDirection extends 'similarity'
    ? TConfig extends NoConfiguration
      ? SimilarityConfiguration
      : TConfig & SimilarityConfiguration
    : TConfig,
  TId
>

interface BuiltInMetricOptions<TDirection extends Direction> {
  readonly implementation: ErasedMetricImplementation & PreparedCapability
  readonly directImplementation?:
    | ((a: MaybeSequence, b: MaybeSequence) => number)
    | undefined
  readonly direction: TDirection
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

function configurationRecord<TDirection extends Direction>(
  configuration: object,
  direction: TDirection,
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

export function builtInMetric<
  TDirection extends Direction,
  TConfig extends object,
  TBrand,
>(options: BuiltInMetricOptions<TDirection>): Metric<TDirection, TConfig, TBrand> {
  // One per metric, shared by every scorer it compiles with default
  // configuration, so choices prepared by one such scorer are accepted by
  // another. A configured scorer gets its own below.
  const defaultPreparedChoiceKey = Object.freeze({})
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
  const compile = (given: TConfig | undefined): MetricCompilation<TDirection, TBrand> => {
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
      indexChoices: preparation.indexChoices,
      proveOptimum: preparation.proveOptimum,
      // `convSequence` copies a string and a plain array-like on its way to a
      // prepared representation, and keeps a typed array by reference. Only
      // that one has to be copied for a handle the caller keeps.
      prepareOwnedChoice: (choice) =>
        preparation.prepareChoice(
          ArrayBuffer.isView(choice) ? snapshotSequence(choice) : choice,
        ),
      preparedChoiceKey: configured ? Object.freeze({}) : defaultPreparedChoiceKey,
    }
  }
  return Object.assign(direct, {
    [COMPILE]: compile,
  })
}
