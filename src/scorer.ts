import {
  customMetricCompilation,
  isBuiltInMetric,
  METRIC_COMPILE,
  type Direction,
  type Metric,
  type MetricCompilation,
  type MissingPolicy,
} from './_metric.js'
import type { MaybeSequence } from './_common.js'

export type { Direction, Metric } from './_metric.js'
export type { MaybeSequence, Sequence } from './_common.js'

export interface ThresholdOptions {
  readonly threshold: number
}

export interface Scorer<D extends Direction = Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  score(a: MaybeSequence, b: MaybeSequence): number
  score(
    a: MaybeSequence,
    b: MaybeSequence,
    options: ThresholdOptions,
  ): number | undefined
}

export interface CustomScorerConfiguration<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly missing?: MissingPolicy | undefined
}

const scorerCompilations = new WeakMap<object, MetricCompilation<Direction>>()

function thresholdOf(options: ThresholdOptions): number {
  if (!Number.isFinite(options.threshold)) {
    throw new RangeError('threshold must be finite')
  }
  return options.threshold
}

function qualifies(direction: Direction, score: number, threshold: number): boolean {
  return direction === 'similarity' ? score >= threshold : score <= threshold
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
    const threshold = thresholdOf(options)
    const lower = compilation.bounds[0]
    const upper = compilation.bounds[1]
    if (
      compilation.trusted &&
      ((compilation.direction === 'similarity' && threshold > upper) ||
        (compilation.direction === 'distance' && threshold < lower))
    ) {
      compilation.validate(a, b)
      return undefined
    }
    const result = compilation.score(a, b, threshold)
    return qualifies(compilation.direction, result, threshold) ? result : undefined
  }
  return score
}

function scorerFromCompilation<D extends Direction>(
  compilation: MetricCompilation<D>,
): Scorer<D> {
  const scorer: Scorer<D> = {
    direction: compilation.direction,
    bounds: Object.freeze([compilation.bounds[0], compilation.bounds[1]]),
    symmetric: compilation.symmetric,
    score: createScoreMethod(compilation),
  }
  scorerCompilations.set(scorer, compilation)
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
    const result: unknown = Reflect.apply(metric[METRIC_COMPILE], metric, [configuration])
    if (!isMetricCompilation<D>(result)) {
      throw new TypeError('built-in metric returned invalid compilation metadata')
    }
    return scorerFromCompilation(result)
  }

  if (!isCustomConfiguration<D>(configuration)) {
    throw new TypeError(
      'custom metrics require direction, bounds, and symmetric configuration',
    )
  }
  validateBounds(configuration.bounds)
  const bounds: readonly [number, number] = Object.freeze([
    configuration.bounds[0],
    configuration.bounds[1],
  ])
  return scorerFromCompilation(
    customMetricCompilation(
      metric,
      configuration.direction,
      bounds,
      configuration.symmetric,
      configuration.missing ?? 'compatible',
    ),
  )
}

function isMetricCompilation<D extends Direction>(
  value: unknown,
): value is MetricCompilation<D> {
  if (typeof value !== 'object' || value === null) return false
  return (
    (Reflect.get(value, 'direction') === 'similarity' ||
      Reflect.get(value, 'direction') === 'distance') &&
    Array.isArray(Reflect.get(value, 'bounds')) &&
    typeof Reflect.get(value, 'symmetric') === 'boolean' &&
    typeof Reflect.get(value, 'validate') === 'function' &&
    typeof Reflect.get(value, 'score') === 'function' &&
    typeof Reflect.get(value, 'prepareQuery') === 'function' &&
    typeof Reflect.get(value, 'prepareChoice') === 'function'
  )
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
    typeof lower !== 'number' ||
    typeof upper !== 'number' ||
    !Number.isFinite(lower) ||
    Number.isNaN(upper) ||
    upper < lower
  ) {
    throw new RangeError('bounds must be an ordered numeric pair with a finite lower bound')
  }
}

export function scorerCompilation(
  scorer: Scorer<Direction>,
): MetricCompilation<Direction> {
  const compilation = scorerCompilations.get(scorer)
  if (compilation === undefined) {
    throw new TypeError('scorer was not created by createScorer')
  }
  if (compilation.direction !== scorer.direction) {
    throw new TypeError('scorer metadata is inconsistent')
  }
  return compilation
}

/** Internal test seam: observes public score access while retaining private kernels. */
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
  scorerCompilations.set(observed, compilation)
  return Object.freeze(observed)
}

export function scoreIfMatch<D extends Direction>(
  scorer: Scorer<D>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): number | undefined {
  return scorer.score(a, b, options)
}

export function isMatch<D extends Direction>(
  scorer: Scorer<D>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): boolean {
  const threshold = thresholdOf(options)
  const compilation = scorerCompilation(scorer)
  if (
    compilation.trusted &&
    ((compilation.direction === 'similarity' && threshold <= compilation.bounds[0]) ||
      (compilation.direction === 'distance' && threshold >= compilation.bounds[1]))
  ) {
    compilation.validate(a, b)
    return true
  }
  return scorer.score(a, b, { threshold }) !== undefined
}
