import { isBuiltInMetric, type Metric } from './metric.js'
import { COMPILE, type MetricCompilation } from './protocol.js'
import { validatePair, validateSequence } from './sequence.js'
import {
  impossibleTrustedThreshold,
  qualifies,
  trustedKernelThreshold,
  validateThreshold,
} from './threshold.js'
import type { Direction, MaybeSequence, MissingPolicy, Sequence } from './types.js'

export interface ThresholdOptions {
  readonly threshold: number
}

export interface Scorer<D extends Direction = Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  score(a: MaybeSequence, b: MaybeSequence): number
  score(a: MaybeSequence, b: MaybeSequence, options: ThresholdOptions): number | undefined
}

export interface CustomScorerConfiguration<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly missing?: MissingPolicy | undefined
}

const compilations = new WeakMap<object, MetricCompilation<Direction>>()

function customCompilation<D extends Direction>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  direction: D,
  bounds: readonly [number, number],
  symmetric: boolean,
  missing: MissingPolicy,
): MetricCompilation<D> {
  const rawScore = (a: Sequence, b: Sequence): number => {
    const result = metric(a, b)
    if (!Number.isFinite(result) || result < bounds[0] || result > bounds[1]) {
      throw new RangeError('custom metric returned a score outside its declared bounds')
    }
    return result
  }
  const score = (a: MaybeSequence, b: MaybeSequence): number => {
    const pair = validatePair(a, b, direction, missing)
    return pair === null ? 0 : rawScore(pair[0], pair[1])
  }
  return {
    direction,
    bounds,
    symmetric,
    trusted: false,
    score: (a, b) => score(a, b),
    rawScore,
    prepareQuery: (query) => (choice) => rawScore(query, validatePreparedChoice(choice)),
    prepareChoice: (choice) => choice,
  }
}

function validatePreparedChoice(value: unknown): Sequence {
  return validateSequence(value)
}

function createScoreMethod<D extends Direction>(
  compilation: MetricCompilation<D>,
): Scorer<D>['score'] {
  function score(a: MaybeSequence, b: MaybeSequence): number
  function score(
    a: MaybeSequence,
    b: MaybeSequence,
    options: ThresholdOptions,
  ): number | undefined
  function score(
    a: MaybeSequence,
    b: MaybeSequence,
    options?: ThresholdOptions,
  ): number | undefined {
    if (options === undefined) return compilation.score(a, b, null)
    const threshold = validateThreshold(options.threshold)
    if (
      compilation.trusted &&
      impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
    ) {
      compilation.validate(a, b)
      return undefined
    }
    const activeThreshold = compilation.trusted
      ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
      : threshold
    const result = compilation.score(a, b, activeThreshold)
    return qualifies(compilation.direction, result, threshold) ? result : undefined
  }
  return score
}

function fromCompilation<D extends Direction>(
  compilation: MetricCompilation<D>,
): Scorer<D> {
  const scorer: Scorer<D> = {
    direction: compilation.direction,
    bounds: Object.freeze([compilation.bounds[0], compilation.bounds[1]]),
    symmetric: compilation.symmetric,
    score: createScoreMethod(compilation),
  }
  compilations.set(scorer, compilation)
  return Object.freeze(scorer)
}

export function createScorer<D extends Direction, Config extends object>(
  metric: Metric<D, Config>,
  configuration?: Config,
): Scorer<D>
export function createScorer<D extends Direction>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  configuration: CustomScorerConfiguration<D>,
): Scorer<D>
export function createScorer<D extends Direction>(
  metric: Metric<D, object> | ((a: MaybeSequence, b: MaybeSequence) => number),
  configuration?: object,
): Scorer<D> {
  if (isBuiltInMetric(metric)) {
    return fromCompilation(metric[COMPILE](configuration))
  }
  if (!isCustomConfiguration<D>(configuration)) {
    throw new TypeError(
      'custom metrics require direction, bounds, and symmetric configuration',
    )
  }
  validateCustomConfigurationKeys(configuration)
  validateBounds(configuration.bounds)
  const missing = configuration.missing ?? 'compatible'
  if (missing !== 'compatible' && missing !== 'throw') {
    throw new TypeError("missing must be 'compatible' or 'throw'")
  }
  if (
    configuration.direction === 'similarity' &&
    missing === 'compatible' &&
    (configuration.bounds[0] > 0 || configuration.bounds[1] < 0)
  ) {
    throw new RangeError(
      "custom similarity bounds must include 0 unless missing is 'throw'",
    )
  }
  const bounds: readonly [number, number] = Object.freeze([
    configuration.bounds[0],
    configuration.bounds[1],
  ])
  return fromCompilation(
    customCompilation(
      metric,
      configuration.direction,
      bounds,
      configuration.symmetric,
      missing,
    ),
  )
}

function validateCustomConfigurationKeys(configuration: object): void {
  for (const key of Object.keys(configuration)) {
    if (
      key !== 'direction' &&
      key !== 'bounds' &&
      key !== 'symmetric' &&
      key !== 'missing'
    ) {
      throw new TypeError(`unknown custom scorer configuration key '${key}'`)
    }
  }
}

function isCustomConfiguration<D extends Direction>(
  value: object | undefined,
): value is CustomScorerConfiguration<D> {
  if (value === undefined) return false
  const direction = Reflect.get(value, 'direction')
  return (
    (direction === 'similarity' || direction === 'distance') &&
    Array.isArray(Reflect.get(value, 'bounds')) &&
    typeof Reflect.get(value, 'symmetric') === 'boolean'
  )
}

function validateBounds(bounds: readonly [number, number]): void {
  const lower = bounds[0]
  const upper = bounds[1]
  if (
    bounds.length !== 2 ||
    typeof lower !== 'number' ||
    typeof upper !== 'number' ||
    !Number.isFinite(lower) ||
    Number.isNaN(upper) ||
    upper < lower
  ) {
    throw new RangeError(
      'bounds must be an ordered numeric pair with a finite lower bound',
    )
  }
}

export function scorerCompilation(
  scorer: Scorer<Direction>,
): MetricCompilation<Direction> {
  const compilation = compilations.get(scorer)
  if (compilation === undefined) {
    throw new TypeError('scorer was not created by createScorer')
  }
  return compilation
}

export function withPublicScoreObserver(
  scorer: Scorer<Direction>,
  observer: () => void,
): Scorer<Direction> {
  const compilation = scorerCompilation(scorer)
  const observed: Scorer<Direction> = {
    direction: scorer.direction,
    bounds: scorer.bounds,
    symmetric: scorer.symmetric,
    get score(): Scorer<Direction>['score'] {
      observer()
      return scorer.score
    },
  }
  compilations.set(observed, compilation)
  return Object.freeze(observed)
}
