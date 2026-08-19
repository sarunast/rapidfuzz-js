import { assertOptionKeys } from '#core/options.js'
import type { AnyMetricCompilation, PreparedKernel } from '#core/scoring/compilation.js'
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
 * - **Indexed search depends on the inner scorer.** Candidate-capable similarity
 *   scorers (currently normalized Indel and exact indexed similarities) let
 *   `createIndexedMatcher` compose exact-token and fuzzy-vocabulary postings.
 *   Every shortlisted choice is still scored by this same Soft Tversky kernel.
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
  readonly nativeThreshold: number

  constructor(
    readonly compilation: AnyMetricCompilation<Direction>,
    readonly threshold: number,
    readonly lower: number,
    readonly span: number,
    upper: number,
  ) {
    this.nativeThreshold = firstQualifyingFloat64(lower, upper, threshold, span)
  }
}

const FLOAT_SIGN = 0x8000_0000_0000_0000n
const FLOAT_MASK = 0xffff_ffff_ffff_ffffn
const floatBuffer = new ArrayBuffer(8)
const floatView = new DataView(floatBuffer)

function floatBits(value: number): bigint {
  floatView.setFloat64(0, value, false)
  return floatView.getBigUint64(0, false)
}

function fromFloatBits(bits: bigint): number {
  floatView.setBigUint64(0, bits, false)
  return floatView.getFloat64(0, false)
}

function floatOrdinal(value: number): bigint {
  const bits = floatBits(value)
  return (bits & FLOAT_SIGN) === 0n ? bits ^ FLOAT_SIGN : bits ^ FLOAT_MASK
}

function fromFloatOrdinal(ordinal: bigint): number {
  const bits = (ordinal & FLOAT_SIGN) === 0n ? ordinal ^ FLOAT_MASK : ordinal ^ FLOAT_SIGN
  return fromFloatBits(bits)
}

/**
 * First representable raw score whose affine normalization reaches threshold.
 *
 * A lower-bound search over the IEEE ordinal keys rather than a `nextUp` walk:
 * the answer can sit 10^17 representable values above `lower`, which no walk
 * reaches. `(raw - lower) / span` is monotone non-decreasing in `raw` because
 * both the subtraction and the division by a positive span are, so the search
 * is exact, and `upper` normalizes to exactly `1` — `span` is `upper - lower` —
 * so a threshold in `(0, 1]` always has an answer.
 */
export function firstQualifyingFloat64(
  lower: number,
  upper: number,
  threshold: number,
  span: number = upper - lower,
): number {
  let low = floatOrdinal(lower)
  let high = floatOrdinal(upper)
  while (low < high) {
    const middle = (low + high) >> 1n
    const raw = fromFloatOrdinal(middle)
    if ((raw - lower) / span >= threshold) high = middle
    else low = middle + 1n
  }
  return fromFloatOrdinal(low)
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
    compilation.bounds[1],
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

/** A leftover as the element scorer sees it: one element, as a string. */
export interface ElementOperand {
  readonly operand: string
}

/**
 * How much comparing a scan must do before the query side is worth preparing.
 *
 * Preparing costs several comparisons, so preparing on first sight loses on
 * every scan too short or too well-matched to earn it back — measured 0.64x
 * against no cache on a single candidate leaving one fuzzy pair, and still
 * 0.95x over sixteen of them.
 *
 * `searchIter`'s `STREAM_PREPARE_AFTER` is the same policy one layer up, and
 * for the same reason: a stream that stops after one candidate must not have
 * paid for a preparation it never used.
 */
const PREPARE_ELEMENT_AFTER = 8

/**
 * The element scorer's query side, prepared once the scan has done enough
 * comparing to earn it, and held for as long as one query is being scanned.
 *
 * A pair scored on its own can never earn it, so the pair paths hold none of
 * this and stay on {@link elementScore}.
 *
 * The choice side is prepared per pair. It could be held on the prepared choice
 * instead and amortized across a matcher's queries, but that would retain a
 * second, opaque representation of every fuzzy-comparable token in the corpus,
 * and that trade needs its own measurement. The array here is refilled per pair
 * and is bounded by the leftovers one pair may offer.
 */
export class ElementKernels {
  readonly #compilation: AnyMetricCompilation<Direction>
  readonly #kernels = new Map<string, PreparedKernel>()
  readonly #columns: unknown[] = []
  #comparisons = 0

  constructor(soft: CompiledElementSimilarity) {
    this.#compilation = soft.compilation
  }

  /**
   * Whether a pair worth `comparisons` should go through prepared kernels,
   * counting it towards the scan either way.
   *
   * Counted in comparisons rather than candidates so that one number serves
   * both shapes a scan is made of: a candidate leaving a full matrix earns the
   * preparation in its first pair or two, and one leaving a single pair waits
   * until the corpus is long enough to be worth it. Before that this is the
   * whole cost of the cache — an add and a compare, no lookup.
   */
  earned(comparisons: number): boolean {
    if (this.#comparisons >= PREPARE_ELEMENT_AFTER) return true
    this.#comparisons += comparisons
    return false
  }

  /** The kernel holding `operand` as its query, prepared on first sight. */
  kernelFor(operand: string): PreparedKernel {
    const held = this.#kernels.get(operand)
    if (held !== undefined) return held
    const kernel = this.#compilation.prepareQuery(operand)
    this.#kernels.set(operand, kernel)
    return kernel
  }

  /** Those elements in the representation the kernels consume, in order. */
  columnsFor(columns: readonly ElementOperand[]): readonly unknown[] {
    this.#columns.length = 0
    for (const column of columns) {
      this.#columns.push(this.#compilation.prepareChoice(column.operand))
    }
    return this.#columns
  }
}

/** {@link elementScore} where the first element is already a prepared kernel. */
export function preparedElementScore(
  soft: CompiledElementSimilarity,
  kernel: PreparedKernel,
  second: unknown,
): number {
  return (kernel(second, null) - soft.lower) / soft.span
}
