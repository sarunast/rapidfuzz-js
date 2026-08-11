import {
  configuredFlagsOf,
  configureOptionsOf,
  isSequence,
  PREPARE_CHOICE,
  prepareScorerOf,
  toRecord,
  type ErasedScorer,
  type MaybeSequence,
  type Sequence,
} from './_common.js'

export type Direction = 'similarity' | 'distance'
export type MissingPolicy = 'compatible' | 'throw'

export interface SimilarityConfiguration {
  readonly missing?: MissingPolicy | undefined
}

export const METRIC_CONFIGURATION: unique symbol = Symbol(
  'rapidfuzz.metric.configuration',
)
export const METRIC_COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

export interface MetricCompilation<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly trusted: boolean
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: (choice: Sequence) => unknown
}

/** A directly callable algorithm carrying private compilation metadata. */
export interface Metric<
  D extends Direction,
  Config extends object = Record<never, never>,
> {
  (a: MaybeSequence, b: MaybeSequence): number
  readonly [METRIC_CONFIGURATION]: (configuration: Config) => Config
  readonly [METRIC_COMPILE]: (
    configuration: Config | undefined,
  ) => MetricCompilation<D>
}

interface BuiltInMetricOptions<D extends Direction, Config extends object> {
  readonly legacy: ErasedScorer
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric?: ((configuration: Readonly<Record<string, unknown>>) => boolean) | undefined
  readonly canonicalize?: ((configuration: Config) => Config) | undefined
}

function assertSequence(value: unknown): Sequence {
  if (!isSequence(value)) {
    throw new TypeError('expected a string or an array-like sequence')
  }
  return value
}

function assertPair(
  a: MaybeSequence,
  b: MaybeSequence,
  direction: Direction,
  missing: MissingPolicy,
): readonly [Sequence, Sequence] | null {
  if (a == null || b == null) {
    if (direction === 'similarity' && missing === 'compatible') return null
    throw new TypeError('missing sequences are not supported by this scorer')
  }
  return [assertSequence(a), assertSequence(b)]
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
    } else {
      record[key] = value
    }
  }
  return { record, missing }
}

function callLegacy(
  legacy: ErasedScorer,
  a: Sequence,
  b: Sequence,
  options: Readonly<Record<string, unknown>> | undefined,
): number {
  const args = options === undefined ? [a, b] : [a, b, options]
  const result: unknown = Reflect.apply(legacy, undefined, args)
  if (typeof result !== 'number') throw new TypeError('metric did not return a number')
  return result
}

function withThreshold(
  configuration: Readonly<Record<string, unknown>>,
  threshold: number,
): Readonly<Record<string, unknown>> {
  return { ...configuration, scoreCutoff: threshold }
}

/** Build a public metric around an existing optimized scorer implementation. */
export function builtInMetric<D extends Direction, Config extends object>(
  options: BuiltInMetricOptions<D, Config>,
): Metric<D, Config> {
  const direct = (a: MaybeSequence, b: MaybeSequence): number => {
    const pair = assertPair(a, b, options.direction, 'compatible')
    return pair === null ? 0 : callLegacy(options.legacy, pair[0], pair[1], undefined)
  }

  const compile = (given: Config | undefined): MetricCompilation<D> => {
    const empty: Record<string, never> = {}
    const canonical =
      given === undefined ? empty : (options.canonicalize?.(given) ?? given)
    const { record: uncanonicalized, missing } = configurationRecord(canonical)
    const legacyCanonicalizer = configureOptionsOf(options.legacy)
    const record =
      legacyCanonicalizer === null
        ? uncanonicalized
        : legacyCanonicalizer(uncanonicalized)
    const flagsResolver = configuredFlagsOf(options.legacy)
    const symmetric =
      options.symmetric?.(record) ?? flagsResolver?.(record).symmetric ?? true
    const prepare = prepareScorerOf(options.legacy)

    const score = (
      a: MaybeSequence,
      b: MaybeSequence,
      threshold: number | null,
    ): number => {
      const pair = assertPair(a, b, options.direction, missing)
      if (pair === null) return 0
      return callLegacy(
        options.legacy,
        pair[0],
        pair[1],
        threshold === null ? record : withThreshold(record, threshold),
      )
    }

    const validate = (a: MaybeSequence, b: MaybeSequence): void => {
      assertPair(a, b, options.direction, missing)
    }

    const prepareQuery = (query: Sequence): PreparedKernel => {
      if (prepare === null) {
        return (choice, threshold) =>
          callLegacy(
            options.legacy,
            query,
            assertSequence(choice),
            threshold === null ? record : withThreshold(record, threshold),
          )
      }
      const prepared = prepare(query, record)
      return (choice, threshold) => prepared(choice, threshold, null)
    }

    const choicePreparer = prepare?.[PREPARE_CHOICE]
    const prepareChoice =
      choicePreparer === undefined
        ? (choice: Sequence): Sequence => choice
        : (choice: Sequence): Sequence | object => {
            const prepared = choicePreparer(choice)
            return typeof prepared === 'object' && prepared !== null ? prepared : choice
          }

    return {
      direction: options.direction,
      bounds: options.bounds,
      symmetric,
      trusted: true,
      validate,
      score,
      prepareQuery,
      prepareChoice,
    }
  }

  return Object.assign(direct, {
    [METRIC_CONFIGURATION]: (configuration: Config): Config => configuration,
    [METRIC_COMPILE]: compile,
  })
}

export function isBuiltInMetric(value: object): value is Metric<Direction, object> {
  return METRIC_COMPILE in value && typeof value[METRIC_COMPILE] === 'function'
}

export function validateSequence(value: unknown): Sequence {
  return assertSequence(value)
}

export function customMetricCompilation<D extends Direction>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  direction: D,
  bounds: readonly [number, number],
  symmetric: boolean,
  missing: MissingPolicy,
): MetricCompilation<D> {
  const score = (a: MaybeSequence, b: MaybeSequence): number => {
    const pair = assertPair(a, b, direction, missing)
    if (pair === null) return 0
    const result = metric(pair[0], pair[1])
    if (!Number.isFinite(result) || result < bounds[0] || result > bounds[1]) {
      throw new RangeError('custom metric returned a score outside its declared bounds')
    }
    return result
  }
  return {
    direction,
    bounds,
    symmetric,
    trusted: false,
    validate: (a, b) => {
      assertPair(a, b, direction, missing)
    },
    score: (a, b) => score(a, b),
    prepareQuery: (query) => (choice) => score(query, assertSequence(choice)),
    prepareChoice: (choice) => choice,
  }
}

export function copyConfiguration(value: unknown): Readonly<Record<string, unknown>> {
  return Object.freeze({ ...toRecord(value) })
}
