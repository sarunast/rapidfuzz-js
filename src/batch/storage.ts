/**
 * Backing storage for the matrix and pairwise scorers.
 *
 * Upstream returns a NumPy array and takes a `dtype`. The port used to return
 * nested plain arrays and keep `dtype` as a flag that only chose whether to
 * round — NumPy's vocabulary attached to a shape that was not a NumPy array.
 * What replaces it is a real typed array, which is what makes the choice worth
 * making: `'u8'` over `'f64'` is eight times less memory for a `ratio` matrix,
 * and the buffer can be handed to a worker or to WebAssembly untouched.
 */

/**
 * Every element type a score can be stored as, by its option value.
 *
 * A type alias rather than an interface, because {@link ScoreArrayKind} is
 * `keyof` it and this one is exported: an interface is open to declaration
 * merging, so a consumer could augment a kind into the key set that
 * `scoreArrayFactory` has no entry for, and get a `RangeError` out of code that
 * type-checked. The kinds are exactly the nine below.
 */
export type ScoreArrayOf = {
  readonly f64: Float64Array
  readonly f32: Float32Array
  readonly i32: Int32Array
  readonly i16: Int16Array
  readonly i8: Int8Array
  readonly u32: Uint32Array
  readonly u16: Uint16Array
  readonly u8: Uint8Array
  readonly u8c: Uint8ClampedArray
}

/**
 * The `into` option: which element type to store scores as, `'f64'` by default.
 *
 * A narrower type is a memory decision — `'u8'` holds any `0..100` fuzz score
 * in an eighth of `'f64'`'s space — and the buffer transfers to a worker or to
 * WebAssembly without copying. A score the type cannot hold is a `RangeError`
 * rather than a silently wrapped number; `'u8c'` is the exception, saturating
 * by definition, and so the way to ask for lossy storage deliberately.
 */
export type ScoreArrayKind = keyof ScoreArrayOf

/** Any of the arrays {@link ScoreArrayOf} names. */
export type ScoreArray = ScoreArrayOf[ScoreArrayKind]

interface ScoreArrayFactory<TArray extends ScoreArray> {
  /** Whether the store holds integers, so a score has to be rounded before it. */
  readonly integral: boolean
  /**
   * The scores the store keeps as written, or `null` when every score is one.
   *
   * `null` for `f64`, which holds any score a JavaScript number can be, and for
   * `u8c`: `Uint8ClampedArray` exists to saturate, so an out-of-range score
   * there is the documented answer rather than a lost one. Every other kind
   * loses a score it cannot hold — the integer kinds wrap, `f32` turns a
   * finite score past its own maximum into `Infinity` — which is what a bound
   * turns into a `RangeError`.
   */
  readonly range: readonly [number, number] | null
  readonly allocate: (length: number) => TArray
  /** A row, sharing the buffer rather than copying it. */
  readonly view: (data: TArray, start: number, end: number) => TArray
}

const FLOAT32_MAX = 3.4028234663852886e38

const SCORE_ARRAYS: {
  readonly [TKind in ScoreArrayKind]: ScoreArrayFactory<ScoreArrayOf[TKind]>
} = {
  f64: {
    range: null,
    integral: false,
    allocate: (n) => new Float64Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  f32: {
    range: [-FLOAT32_MAX, FLOAT32_MAX],
    integral: false,
    allocate: (n) => new Float32Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  i32: {
    range: [-(2 ** 31), 2 ** 31 - 1],
    integral: true,
    allocate: (n) => new Int32Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  i16: {
    range: [-32768, 32767],
    integral: true,
    allocate: (n) => new Int16Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  i8: {
    range: [-128, 127],
    integral: true,
    allocate: (n) => new Int8Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  u32: {
    range: [0, 2 ** 32 - 1],
    integral: true,
    allocate: (n) => new Uint32Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  u16: {
    range: [0, 65535],
    integral: true,
    allocate: (n) => new Uint16Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  u8: {
    range: [0, 255],
    integral: true,
    allocate: (n) => new Uint8Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  u8c: {
    range: null,
    integral: true,
    allocate: (n) => new Uint8ClampedArray(n),
    view: (d, s, e) => d.subarray(s, e),
  },
}

/** The factory for a kind, with its element type carried through. */
export function scoreArrayFactory<TKind extends ScoreArrayKind>(
  kind: TKind,
): ScoreArrayFactory<ScoreArrayOf[TKind]> {
  const factory = SCORE_ARRAYS[kind]
  if (factory === undefined) {
    throw new RangeError(`unknown score array kind: ${String(kind)}`)
  }
  return factory
}

const MAX_LENGTH = 2 ** 32 - 1

/**
 * Allocate `length` elements of `kind`, refusing a length no array can hold.
 *
 * The two checks are in this order because they answer different questions: a
 * `queries × choices` product genuinely reaches the first from ordinary arrays,
 * where the second is only reachable from JavaScript passing something that is
 * not an array at all — and a typed-array constructor would coerce `1.5` to a
 * one-element array rather than say so.
 */
export function allocateScores<TKind extends ScoreArrayKind>(
  kind: TKind,
  length: number,
  what: string,
): ScoreArrayOf[TKind] {
  if (length > MAX_LENGTH) {
    throw new RangeError(`${what} needs ${length} scores, more than an array can hold`)
  }
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError(`${what} needs ${length} scores, which is not a count`)
  }
  return scoreArrayFactory(kind).allocate(length)
}

/**
 * Round half away from zero, which is what an integral dtype gets upstream.
 *
 * `Math.round` breaks a tie towards positive infinity, so it agrees on every
 * non-negative score and disagrees on `-0.5`, `-1.5`, `-2.5` — reachable
 * through a negative `scoreMultiplier`, since no scorer returns a negative
 * score itself. An integer typed array applies its own conversion on assignment
 * — truncating for most of them, clamping and rounding to even for
 * `Uint8ClampedArray` — and none of those is this rule, so leaving the rounding
 * to the store would silently replace it.
 *
 * The zero is normalised because an integer array upstream holds `0` where
 * rounding a small negative score here would otherwise leave JavaScript's `-0`.
 * An integer destination erases the sign on its own; a float one does not,
 * which is why this still matters.
 */
export function roundHalfAwayFromZero(value: number): number {
  const rounded = value < 0 ? -Math.round(-value) : Math.round(value)
  return rounded === 0 ? 0 : rounded
}

/**
 * The bound every stored score has to be tested against, or `null` when none
 * does.
 *
 * A store that cannot hold a score does not say so: an integer typed array
 * wraps `300` written into a `u8` to `44`, which reads as a score. Refusing the
 * *kind* up front on the scorer's bounds alone would be far too coarse — a
 * Levenshtein distance is bounded by `Infinity` and is almost always a number a
 * `u8` holds — so the range is proven where it can be and tested where it
 * cannot.
 *
 * The proof is the two ends of `bounds`, scaled and rounded exactly as a score
 * is: rounding is monotone, so the extremes stay the extremes. `Infinity`, a
 * `NaN` from `Infinity * 0`, and anything else that fails a comparison fall
 * through to the tested path rather than out of it. What the proof rests on is
 * the scorer's own declaration — a metric returning scores outside the bounds
 * it states is unchecked here, as it is everywhere else.
 */
export function scoreStoreRange(
  kind: ScoreArrayKind,
  bounds: readonly [number, number],
  multiplier: number,
): readonly [number, number] | null {
  const { range, integral } = scoreArrayFactory(kind)
  if (range === null) return null
  const scaledLow = Math.min(bounds[0] * multiplier, bounds[1] * multiplier)
  const scaledHigh = Math.max(bounds[0] * multiplier, bounds[1] * multiplier)
  const low = integral ? roundHalfAwayFromZero(scaledLow) : scaledLow
  const high = integral ? roundHalfAwayFromZero(scaledHigh) : scaledHigh
  return low >= range[0] && high <= range[1] ? null : range
}

/** The refusal {@link scoreStoreRange}'s bound leads to, worded once. */
export function unstorableScore(
  value: number,
  kind: ScoreArrayKind,
  what: string,
): never {
  throw new RangeError(
    `${what} produced the score ${value}, which '${kind}' cannot store: score ` +
      `into a wider element type, or into 'u8c' to saturate instead`,
  )
}
