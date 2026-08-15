import { snapshotSequence, validatePair, validateSequence } from '../../sequence.js'
import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Sequence,
  SimilarityConfiguration,
} from '../../types.js'
import { COMPILE, type MetricCompilation } from '../compilation.js'
import type { Metric, NoConfiguration } from '../metric.js'
import {
  configurationCanonicalizerOf,
  configurationSymmetryOf,
  type ErasedMetricImplementation,
  type PreparedCapability,
} from './implementation.js'
import type { ScorerOptions } from './options.js'
import { PREPARE_SCORER } from './preparation.js'

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
  const defaultPreparedChoiceKey = Object.freeze({})
  const implementation = options.implementation
  const direct =
    options.directImplementation ??
    ((a: MaybeSequence, b: MaybeSequence): number => {
      if (a == null || b == null) {
        if (options.direction === 'similarity') return 0
        throw new TypeError('missing sequences are not supported by this scorer')
      }
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
