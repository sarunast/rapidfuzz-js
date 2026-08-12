import { isBuiltInMetric, type Metric } from './metric.js'
import { createPreparedChoice, type AnyBrand, type PreparedChoice } from './prepared.js'
import { COMPILE, type MetricCompilation } from './protocol.js'
import { snapshotSequence, validatePair, validateSequence } from './sequence.js'
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

export interface Scorer<D extends Direction = Direction, Brand = AnyBrand> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  score(a: MaybeSequence, b: MaybeSequence): number
  score(a: MaybeSequence, b: MaybeSequence, options: ThresholdOptions): number | undefined
  prepareChoice(choice: Sequence): PreparedChoice<Brand>
}

export type PreparedChoiceOf<S extends { prepareChoice: (choice: never) => unknown }> =
  ReturnType<S['prepareChoice']>

// What `createScorer(metric)` infers, nameable from a metric alone — for the
// annotation on a stored scorer that should keep its metric's brand.
export type ScorerOf<M> =
  M extends Metric<infer D extends Direction, infer _Config extends object, infer Brand>
    ? Scorer<D, Brand>
    : never

export interface CustomScorerConfiguration<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly missing?: MissingPolicy | undefined
}

const compilations = new WeakMap<object, MetricCompilation<Direction>>()

function customCompilation<D extends Direction, B>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  direction: D,
  bounds: readonly [number, number],
  symmetric: boolean,
  missing: MissingPolicy,
): MetricCompilation<D, B> {
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
    // Passed rather than wrapped: a two-parameter function satisfies the
    // three-parameter shape, and a custom metric has no cutoff to take.
    score,
    rawScore,
    prepareQuery: (query) => (choice) => rawScore(query, validatePreparedChoice(choice)),
    prepareChoice: (choice) => choice,
    // A custom metric is handed the sequence itself, so a handle that outlives
    // the call needs a copy of its own.
    prepareOwnedChoice: (choice) => snapshotSequence(choice),
    // Fresh per call: a custom metric is whatever the caller passed, so two
    // scorers built from one function still prepare choices for themselves.
    preparedChoiceKey: Object.freeze({}),
  }
}

function validatePreparedChoice(value: unknown): Sequence {
  return validateSequence(value)
}

function createScoreMethod<D extends Direction>(
  compilation: MetricCompilation<D>,
): Scorer<D, never>['score'] {
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

function fromCompilation<D extends Direction, B>(
  compilation: MetricCompilation<D, B>,
): Scorer<D, B> {
  const scorer: Scorer<D, B> = {
    direction: compilation.direction,
    bounds: Object.freeze([compilation.bounds[0], compilation.bounds[1]]),
    symmetric: compilation.symmetric,
    score: createScoreMethod(compilation),
    // Owned, not borrowed: a handle outlives this call, so mutating the
    // sequence afterwards must not reach through it. `createMatcher` snapshots
    // for the same reason, and the two have to agree.
    prepareChoice: (choice) =>
      createPreparedChoice(
        compilation.preparedChoiceKey,
        compilation.prepareOwnedChoice(validateSequence(choice)),
      ),
  }
  compilations.set(scorer, compilation)
  return Object.freeze(scorer)
}

export function createScorer<D extends Direction, Config extends object, B>(
  metric: Metric<D, Config, B>,
  configuration: Config,
): Scorer<D, B>
// Without a configuration there is nothing to infer `Config` from, and trying
// to is what refused a union of metrics whose configurations have no key in
// common — `levenshtein.distance` beside `jaroWinkler.distance`, as a loop over
// an array of metrics produces.
export function createScorer<D extends Direction, B>(
  metric: Metric<D, never, B>,
  configuration?: undefined,
): Scorer<D, B>
// A metric whose brand cannot be pinned still compiles a scorer; what it gives
// up is the compile-time half of the prepared-choice check.
export function createScorer<D extends Direction>(
  metric: Metric<D, never, AnyBrand>,
  configuration?: undefined,
): Scorer<D>
export function createScorer<D extends Direction>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  configuration: CustomScorerConfiguration<D>,
): Scorer<D>
// `Metric<D, never>` is "whatever its configuration", not "it takes none": the
// compile hook is contravariant, so `object` would demand a hook accepting any
// object and no built-in would be assignable. The overloads keep the real
// `Config`; this line only has to admit them all.
export function createScorer<D extends Direction, B>(
  metric: Metric<D, never, B> | ((a: MaybeSequence, b: MaybeSequence) => number),
  configuration?: object,
): Scorer<D, B> {
  // The direction is named rather than inferred: the guard reads `unknown`, so
  // there is no argument left to infer `D` from, and a bare call would widen the
  // result to `Scorer<Direction>`.
  if (isBuiltInMetric<D, object, B>(metric)) {
    return fromCompilation(metric[COMPILE](configuration))
  }
  // The guard above answers `false` for a non-callable rather than throwing, so
  // refuse it here — otherwise the scorer builds and fails at first use.
  if (typeof metric !== 'function') {
    throw new TypeError('metric must be a function')
  }
  if (!isCustomConfiguration<D>(configuration)) {
    throw new TypeError(
      'custom metrics require direction, bounds, and symmetric configuration',
    )
  }
  validateCustomConfigurationKeys(configuration, configuration.direction)
  validateBounds(configuration.bounds)
  // Only similarity has a policy to choose: `validatePair` throws for a missing
  // side of a distance pair whatever it is told.
  const missing: MissingPolicy =
    configuration.direction === 'similarity'
      ? (configuration.missing ?? 'compatible')
      : 'throw'
  if (missing !== 'compatible' && missing !== 'throw') {
    throw new TypeError("missing must be 'compatible' or 'throw'")
  }
  if (
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
    customCompilation<D, B>(
      metric,
      configuration.direction,
      bounds,
      configuration.symmetric,
      missing,
    ),
  )
}

function validateCustomConfigurationKeys(
  configuration: object,
  direction: Direction,
): void {
  for (const key of Object.keys(configuration)) {
    // `missing` is similarity-only, as it is for a built-in metric — see
    // `configurationRecord` in `algorithms/shared/metricAdapter`.
    const known =
      key === 'direction' ||
      key === 'bounds' ||
      key === 'symmetric' ||
      (key === 'missing' && direction === 'similarity')
    if (!known) {
      throw new TypeError(`unknown custom scorer configuration key '${key}'`)
    }
  }
}

// Reads `unknown` because a JavaScript caller's `null` or `123` gets here, and
// `Reflect.get` on a non-object throws an error about our internals.
function isCustomConfiguration<D extends Direction>(
  value: unknown,
): value is CustomScorerConfiguration<D> {
  if (typeof value !== 'object' || value === null) return false
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
    // Copied plainly: the observer reports public `score` calls, and preparing
    // a choice is not one.
    prepareChoice: (choice) => scorer.prepareChoice(choice),
  }
  compilations.set(observed, compilation)
  return Object.freeze(observed)
}
