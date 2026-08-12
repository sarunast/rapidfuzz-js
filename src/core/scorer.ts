import { isBuiltInMetric, type Metric } from './metric.js'
import { assertOptionKeys } from './options.js'
import { createPreparedChoice, type AnyBrand, type PreparedChoice } from './prepared.js'
import { COMPILE, type MetricCompilation } from './protocol.js'
import {
  normalizeSequence,
  snapshotSequence,
  validatePair,
  validateSequence,
} from './sequence.js'
import {
  impossibleTrustedThreshold,
  qualifies,
  trustedKernelThreshold,
  validateThreshold,
} from './threshold.js'
import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Normalizer,
  Sequence,
} from './types.js'

/**
 * The quality bar a score has to clear, on the scorer's own scale — `0..100`
 * for a fuzz scorer, `0..1` for a normalized one, a count for a raw distance.
 *
 * A similarity reads it as a minimum, a distance as a maximum. It is also a
 * speed lever: the edit-distance kernels use it as a cutoff and abandon a pair
 * as soon as it can no longer qualify.
 */
export interface ThresholdOptions {
  readonly threshold: number
}

/** Accepted by {@link Scorer.prepareChoice}. */
export interface PrepareChoiceOptions {
  /**
   * Applied to the choice before it is prepared. The search that later scores
   * the handle has to name the same function, by identity — a handle records
   * which normalizer made it, so the two sides cannot be compared having been
   * cleaned differently.
   */
  readonly normalize?: Normalizer | undefined
}

const PREPARE_CHOICE_OPTION_KEYS = [
  'normalize',
] as const satisfies readonly (keyof PrepareChoiceOptions)[]

/**
 * A metric with its decisions attached — configuration, missing-value policy —
 * made once and applied to every comparison. Everything that searches takes a
 * scorer rather than a bare metric, so the search machinery never has to guess
 * how comparisons should be made.
 *
 * Built by {@link createScorer}, and frozen.
 */
export interface Scorer<TDirection extends Direction = Direction, TBrand = AnyBrand> {
  /** `'similarity'` (higher is better) or `'distance'` (lower is better). */
  readonly direction: TDirection
  /** The inclusive range every score falls in, as `[minimum, maximum]`. */
  readonly bounds: readonly [number, number]
  /** Whether `score(a, b) === score(b, a)` for every pair. */
  readonly symmetric: boolean
  /**
   * Score one pair.
   *
   * @throws `TypeError` if either operand is neither a string nor an array-like
   * sequence, or is missing where the scorer's policy refuses it — a distance
   * scorer always refuses, a similarity scorer reports `0` unless built with
   * `missing: 'throw'`.
   */
  score(a: MaybeSequence, b: MaybeSequence): number
  /**
   * Score one pair, or `undefined` when it does not clear `threshold`.
   *
   * `undefined` rather than `0` or `-1` because `0` is a legitimate score, so
   * no sentinel could be trusted.
   */
  score(a: MaybeSequence, b: MaybeSequence, options: ThresholdOptions): number | undefined
  /**
   * Convert one candidate into the form this scorer's kernels want, returning
   * an opaque handle to store beside your own data and hand back through a
   * search's `getPrepared`. Repeated searches then prepare nothing.
   *
   * The handle copies what it holds, so mutating the source sequence afterwards
   * does not reach through it.
   *
   * @throws `TypeError` if `choice` is not a valid sequence.
   */
  prepareChoice(choice: Sequence, options?: PrepareChoiceOptions): PreparedChoice<TBrand>
}

/**
 * The handle type a given scorer produces — for annotating a stored handle
 * without naming the metric's brand by hand:
 *
 * ```ts
 * interface Row {
 *   prepared: PreparedChoiceOf<typeof scorer>
 * }
 * ```
 */
export type PreparedChoiceOf<
  TScorer extends { prepareChoice: (choice: never) => unknown },
> = ReturnType<TScorer['prepareChoice']>

/**
 * What `createScorer(metric)` infers, nameable from a metric alone — for the
 * annotation on a stored scorer that should keep its metric's brand.
 */
export type ScorerOf<TMetric> =
  TMetric extends Metric<
    infer TDirection extends Direction,
    infer _TConfig extends object,
    infer TBrand
  >
    ? Scorer<TDirection, TBrand>
    : never

/**
 * What a plain `(a, b) => number` has to declare to join the system. The
 * library enforces every field: a result outside `bounds`, or not finite, is a
 * `RangeError` raised before any thresholding or ordering can rely on it.
 */
export interface CustomScorerConfiguration<TDirection extends Direction> {
  /** Whether higher scores are better (`'similarity'`) or worse (`'distance'`). */
  readonly direction: TDirection
  /** The inclusive `[minimum, maximum]` every score must fall inside. */
  readonly bounds: readonly [number, number]
  /** Whether `f(a, b) === f(b, a)`, which lets callers reuse work across a pair. */
  readonly symmetric: boolean
  /**
   * What a `null`/`undefined` operand means. `'compatible'` (the default)
   * reports the worst score; `'throw'` treats it as a bug upstream. Distance
   * scorers always throw — there is no honest "worst possible distance" when
   * distances are unbounded — so this only has an effect on a similarity.
   */
  readonly missing?: MissingPolicy | undefined
}

const compilations = new WeakMap<object, MetricCompilation<Direction>>()

function customCompilation<TDirection extends Direction, TBrand>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  direction: TDirection,
  bounds: readonly [number, number],
  symmetric: boolean,
  missing: MissingPolicy,
): MetricCompilation<TDirection, TBrand> {
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

function createScoreMethod<TDirection extends Direction>(
  compilation: MetricCompilation<TDirection>,
): Scorer<TDirection, never>['score'] {
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

function fromCompilation<TDirection extends Direction, TBrand>(
  compilation: MetricCompilation<TDirection, TBrand>,
): Scorer<TDirection, TBrand> {
  const scorer: Scorer<TDirection, TBrand> = {
    direction: compilation.direction,
    bounds: Object.freeze([compilation.bounds[0], compilation.bounds[1]]),
    symmetric: compilation.symmetric,
    score: createScoreMethod(compilation),
    // Owned, not borrowed: a handle outlives this call, so mutating the
    // sequence afterwards must not reach through it. `createMatcher` snapshots
    // for the same reason, and the two have to agree.
    prepareChoice: (choice, options) => {
      // Guarded rather than checked unconditionally: this runs once per choice,
      // and the call that names no options has no keys to walk.
      if (options !== undefined) {
        assertOptionKeys(options, PREPARE_CHOICE_OPTION_KEYS, 'prepareChoice')
      }
      const valid = validateSequence(choice)
      const normalize = options?.normalize
      if (normalize === undefined) {
        return createPreparedChoice(
          compilation.preparedChoiceKey,
          compilation.prepareOwnedChoice(valid),
          undefined,
        )
      }
      if (typeof normalize !== 'function') {
        throw new TypeError('normalize must be a function')
      }
      return createPreparedChoice(
        compilation.preparedChoiceKey,
        compilation.prepareOwnedChoice(normalizeSequence(valid, normalize)),
        normalize,
      )
    },
  }
  compilations.set(scorer, compilation)
  return Object.freeze(scorer)
}

/**
 * Build a {@link Scorer} from a built-in metric and its configuration.
 *
 * A scorer is a metric with its decisions already made — how the algorithm is
 * tuned, what happens to missing values — so the same decisions apply to every
 * comparison and everything that searches can take one without guessing.
 *
 * The configuration is type-checked against that specific metric, so passing
 * `weights` to a metric that has none is a compile error rather than a silent
 * no-op.
 *
 * ```ts
 * const weighted = createScorer(levenshtein.distance, {
 *   weights: { insertion: 1, deletion: 1, substitution: 2 },
 * })
 * weighted.score('kitten', 'sitting') // 5
 * weighted.score('kitten', 'sitting', { threshold: 3 }) // undefined
 * ```
 *
 * Scorers are frozen and stateless, so build them once at module scope and
 * share them. Two scorers built from the same metric with its default
 * preparation also share prepared choices.
 *
 * @param metric A built-in metric from an algorithm subpath.
 * @param configuration That metric's own options, if it takes any.
 * @returns A frozen {@link Scorer}, branded with the metric so its prepared
 * choices are checked at compile time as well as at runtime.
 */
export function createScorer<
  TDirection extends Direction,
  TConfig extends object,
  TBrand,
>(
  metric: Metric<TDirection, TConfig, TBrand>,
  configuration: TConfig,
): Scorer<TDirection, TBrand>
// Without a configuration there is nothing to infer `Config` from, and trying
// to is what refused a union of metrics whose configurations have no key in
// common — `levenshtein.distance` beside `jaroWinkler.distance`, as a loop over
// an array of metrics produces.
export function createScorer<TDirection extends Direction, TBrand>(
  metric: Metric<TDirection, never, TBrand>,
  configuration?: undefined,
): Scorer<TDirection, TBrand>
// A metric whose brand cannot be pinned still compiles a scorer; what it gives
// up is the compile-time half of the prepared-choice check.
export function createScorer<TDirection extends Direction>(
  metric: Metric<TDirection, never, AnyBrand>,
  configuration?: undefined,
): Scorer<TDirection>
/**
 * Build a {@link Scorer} from your own `(a, b) => number` function, which then
 * works with thresholds, ranking, searches and Matchers like any built-in.
 *
 * ```ts
 * const exact = createScorer((a, b) => (a === b ? 1 : 0), {
 *   direction: 'similarity',
 *   bounds: [0, 1],
 *   symmetric: true,
 * })
 * ```
 *
 * @throws `TypeError` if `metric` is not a function, or the configuration omits
 * `direction`, `bounds` or `symmetric`.
 * @throws `RangeError` if `bounds` is not an ordered numeric pair.
 */
export function createScorer<TDirection extends Direction>(
  metric: (a: MaybeSequence, b: MaybeSequence) => number,
  configuration: CustomScorerConfiguration<TDirection>,
): Scorer<TDirection>
// `Metric<TDirection, never>` is "whatever its configuration", not "it takes none": the
// compile hook is contravariant, so `object` would demand a hook accepting any
// object and no built-in would be assignable. The overloads keep the real
// `Config`; this line only has to admit them all.
export function createScorer<TDirection extends Direction, TBrand>(
  metric:
    | Metric<TDirection, never, TBrand>
    | ((a: MaybeSequence, b: MaybeSequence) => number),
  configuration?: object,
): Scorer<TDirection, TBrand> {
  // The direction is named rather than inferred: the guard reads `unknown`, so
  // there is no argument left to infer `D` from, and a bare call would widen the
  // result to `Scorer<Direction>`.
  if (isBuiltInMetric<TDirection, object, TBrand>(metric)) {
    return fromCompilation(metric[COMPILE](configuration))
  }
  // The guard above answers `false` for a non-callable rather than throwing, so
  // refuse it here — otherwise the scorer builds and fails at first use.
  if (typeof metric !== 'function') {
    throw new TypeError('metric must be a function')
  }
  if (!isCustomConfiguration<TDirection>(configuration)) {
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
    customCompilation<TDirection, TBrand>(
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
function isCustomConfiguration<TDirection extends Direction>(
  value: unknown,
): value is CustomScorerConfiguration<TDirection> {
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
    prepareChoice: (choice, options) => scorer.prepareChoice(choice, options),
  }
  compilations.set(observed, compilation)
  return Object.freeze(observed)
}
