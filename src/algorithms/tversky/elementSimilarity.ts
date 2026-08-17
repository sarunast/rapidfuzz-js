import { assertOptionKeys } from '#core/options.js'
import type { AnyMetricCompilation } from '#core/scoring/compilation.js'
import { scorerCompilation, type Scorer } from '#core/scoring/scorer.js'
import type { Direction } from '#core/types.js'

/**
 * Fuzzy element matching: an inner scorer, and how alike two elements must be
 * before they may share any mass at all.
 *
 * Exact overlap still decides everything it can. Only the occurrences exact
 * matching left over are offered to `scorer`, and each surviving pair shares
 * `min(firstWeight, secondWeight) × similarity` rather than a whole occurrence.
 * So `['swisscom', 'ag']` against `['swisscomm', 'ag']` pairs `ag` with `ag`
 * exactly, then `swisscom` with `swisscomm` partially.
 *
 * ```ts
 * const company = createScorer(tverskySimilarity, {
 *   gramSize: 1,
 *   elementSimilarity: {
 *     scorer: createScorer(indelNormalizedSimilarity),
 *     threshold: 0.8,
 *   },
 * })
 * ```
 *
 * Five traps, in the order they bite:
 *
 * - **Only multi-character string tokens are compared.** A single code point
 *   canonicalizes to a number — astral characters included — so `['a']` and
 *   `['😀']` are exact-only, and a plain string at `gramSize: 1` is a sequence
 *   of code points and therefore scores exactly what it scores today. This is a
 *   token-array feature.
 * - **Exact pairs are reserved first**, so the result is the best matching over
 *   what exact matching left, not the best matching overall. `['google', 'x']`
 *   against `['google', 'y']` always pairs `google` with `google`, even where
 *   pairing it with `y` would have scored higher. "Best" is also subject to
 *   floating-point path arithmetic — see `maximumTransport` — so two matchings
 *   whose totals differ in the last bit are not distinguished.
 * - **It costs `n × m` element comparisons** on the distinct unmatched elements,
 *   followed by a transport solve. Past 32 distinct *fuzzy-comparable* leftovers
 *   on either side it throws rather than quietly becoming slow — leftovers no
 *   element scorer can see do not count, and neither side counts at all when the
 *   other has none. The limit is per side because the solve, not the comparing,
 *   is what runs away on a long sequence against a short one. Repeats do not
 *   count either, though they are not quite free: skewing how often elements
 *   repeat costs the solve more paths, and a separate ceiling refuses a pair that
 *   takes more than 512 of them.
 * - **A soft scorer has no indexed representation**, so `createIndexedMatcher`
 *   refuses it; the index scores exact overlap and would disagree.
 * - **`symmetric` is `false`.** The optimum itself is symmetric where
 *   `alpha === beta`, but tie-breaking and fold order are not transpose
 *   invariant, so the last bit may differ.
 *
 * `scorer` and `threshold` are captured when the outer scorer is created;
 * mutating this object afterwards changes nothing.
 *
 * @throws {TypeError} If it is not an object, holds an unknown key, or `scorer`
 *   is not a similarity scorer from `createScorer`, is asymmetric, or
 *   `threshold` is not a number.
 * @throws {RangeError} If `gramSize` is not `1`, `threshold` is outside
 *   `0 < threshold <= 1`, or the scorer's `bounds` do not span a finite,
 *   non-zero range.
 */
export interface TverskyElementSimilarity {
  /**
   * How alike two elements are, on its own scale — a scorer bounded `0..100`
   * is rescaled onto `0..1` for you, so the fuzz scorers work unadapted.
   *
   * It must be a symmetric similarity scorer with finite bounds spanning a
   * finite, non-zero range, and it must be deterministic: the library cannot
   * enforce that of a custom scoring function, and a stateful one makes
   * `score` and `explain` disagree.
   */
  readonly scorer: Scorer<'similarity'>
  /**
   * The least similarity, on `0..1`, that lets two elements share mass.
   *
   * Above `0` and at most `1`. There is deliberately no default — a useful
   * threshold is a property of the data, and a library one would be a wrong
   * number with an authoritative air. `0` is refused: it admits arbitrarily
   * weak pairings and makes the matching dense.
   */
  readonly threshold: number
}

/**
 * What an `elementSimilarity` option becomes once the scorer is created: the
 * inner metric's compilation, the threshold, and the affine constants that
 * carry a raw score onto `0..1`.
 *
 * Compiled here rather than read per pair so a configured scorer validates once
 * and cannot observe a later mutation of the caller's option object.
 */
export class CompiledElementSimilarity {
  constructor(
    readonly compilation: AnyMetricCompilation<Direction>,
    readonly threshold: number,
    readonly lower: number,
    readonly span: number,
  ) {}
}

const OPTION_KEYS: readonly string[] = ['scorer', 'threshold']

function isScorerLike(value: unknown): value is Scorer<Direction> {
  if (typeof value !== 'object' || value === null) return false
  const direction = Reflect.get(value, 'direction')
  return (
    (direction === 'similarity' || direction === 'distance') &&
    Array.isArray(Reflect.get(value, 'bounds')) &&
    typeof Reflect.get(value, 'symmetric') === 'boolean' &&
    typeof Reflect.get(value, 'score') === 'function' &&
    typeof Reflect.get(value, 'prepareChoice') === 'function'
  )
}

function validThreshold(value: unknown): number {
  if (typeof value !== 'number') {
    throw new TypeError('elementSimilarity.threshold must be a number')
  }
  if (!(value > 0 && value <= 1)) {
    throw new RangeError('elementSimilarity.threshold has to be above 0 and at most 1')
  }
  return value
}

/**
 * The affine constants that rescale the inner scorer onto `0..1`.
 *
 * The span carries the whole contract, which is why the endpoints are not
 * checked separately: an infinite endpoint makes it infinite, a `NaN` one makes
 * it `NaN`, and equal endpoints make it zero. Two *finite* endpoints are not
 * enough on their own — `[-MAX_VALUE, MAX_VALUE]` overflows to `Infinity` and
 * would rescale every score to `0`. Core accepts all of those.
 */
function validSpan(bounds: readonly [number, number]): number {
  const span = bounds[1] - bounds[0]
  if (!Number.isFinite(span) || span <= 0) {
    throw new RangeError(
      'elementSimilarity.scorer needs finite bounds spanning a finite, non-zero range',
    )
  }
  return span
}

/**
 * Element similarity asks for exact element overlap to be *completed*, which is
 * what `gramSize: 1` is: a shingle of several elements is not a thing a caller
 * named, and has no single similarity to another shingle.
 */
export function compileElementSimilarity(
  raw: unknown,
  gramSize: number,
): CompiledElementSimilarity {
  if (gramSize !== 1) {
    throw new RangeError('element similarity is only defined at gramSize 1')
  }
  if (raw instanceof CompiledElementSimilarity) return raw
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError(
      'elementSimilarity must be an object with a scorer and a threshold',
    )
  }
  assertOptionKeys(raw, OPTION_KEYS, 'elementSimilarity')
  const scorer = Reflect.get(raw, 'scorer')
  if (!isScorerLike(scorer)) {
    throw new TypeError('elementSimilarity.scorer must be a scorer from createScorer')
  }
  const threshold = validThreshold(Reflect.get(raw, 'threshold'))
  const compilation = scorerCompilation(scorer)
  if (compilation.direction !== 'similarity') {
    throw new TypeError('elementSimilarity.scorer must be a similarity scorer')
  }
  if (!compilation.symmetric) {
    throw new TypeError('elementSimilarity.scorer must be symmetric')
  }
  const span = validSpan(compilation.bounds)
  return new CompiledElementSimilarity(
    compilation,
    threshold,
    compilation.bounds[0],
    span,
  )
}

/** The compiled option, or `null` where the caller asked for exact matching. */
export function effectiveElementSimilarity(
  raw: unknown,
  gramSize: number,
): CompiledElementSimilarity | null {
  return raw === undefined ? null : compileElementSimilarity(raw, gramSize)
}

/**
 * How alike two elements are, on `0..1`.
 *
 * The result lands in `[0, 1]` with no clamp, and the soft fold depends on that
 * being exact rather than approximate — every residual it forms is
 * `weight − min(wa, wb) · s`, which is non-negative in IEEE only while `s ≤ 1`
 * is. A raw score is always finite and within the declared bounds, since a
 * custom metric that leaves them throws before returning here. Subtraction from
 * a fixed operand and division by a fixed one are both monotone under
 * round-to-nearest, so `raw ≤ upper` gives `raw − lower ≤ span` gives
 * `(raw − lower) / span ≤ span / span = 1`, and `raw ≥ lower` gives `≥ 0`.
 */
export function elementScore(
  soft: CompiledElementSimilarity,
  first: string,
  second: string,
): number {
  return (soft.compilation.rawScore(first, second, null) - soft.lower) / soft.span
}
