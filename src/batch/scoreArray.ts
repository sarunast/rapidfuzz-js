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

/** The `into` option: which element type to store scores as. */
export type ScoreArrayKind = keyof ScoreArrayOf

/** Any of the arrays {@link ScoreArrayOf} names. */
export type ScoreArray = ScoreArrayOf[ScoreArrayKind]

/**
 * Function properties rather than methods, so `A` is checked contravariantly
 * under `strictFunctionTypes` — a method declaration is bivariant, which would
 * let a `Float64Array` factory sit in the `u8` slot of the table below.
 */
interface ScoreArrayFactory<A extends ScoreArray> {
  /** Whether the store holds integers, so a score has to be rounded before it. */
  readonly integral: boolean
  /**
   * The scores the store keeps as written, or `null` when every score is one.
   *
   * `null` for the two float kinds, which hold any score a metric can produce,
   * and for `u8c`: `Uint8ClampedArray` exists to saturate, so an out-of-range
   * score there is the documented answer rather than a lost one. Every other
   * kind wraps silently, which is what a bound turns into a `RangeError`.
   */
  readonly range: readonly [number, number] | null
  readonly allocate: (length: number) => A
  /** A row, sharing the buffer rather than copying it. */
  readonly view: (data: A, start: number, end: number) => A
}

/**
 * One concrete factory per kind.
 *
 * Written out rather than derived from a constructor the caller passes, because
 * `new into(n)` for a generic `into` resolves its construct signature from the
 * *constraint* and comes back as the whole union — recovering `Uint8Array` from
 * it needs an assertion, which this project does not allow. Indexing a mapped
 * type by the generic key propagates the instantiation instead, and each entry
 * here is concrete so `subarray` has a single unambiguous signature.
 *
 * A static table over the built-in typed arrays: nothing registers a kind at
 * runtime, and no constructor has to be abstracted over.
 */
const SCORE_ARRAYS: {
  readonly [K in ScoreArrayKind]: ScoreArrayFactory<ScoreArrayOf[K]>
} = {
  f64: {
    range: null,
    integral: false,
    allocate: (n) => new Float64Array(n),
    view: (d, s, e) => d.subarray(s, e),
  },
  f32: {
    range: null,
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
export function scoreArrayFactory<K extends ScoreArrayKind>(
  kind: K,
): ScoreArrayFactory<ScoreArrayOf[K]> {
  const factory = SCORE_ARRAYS[kind]
  if (factory === undefined) {
    throw new RangeError(`unknown score array kind: ${String(kind)}`)
  }
  return factory
}

/**
 * The largest score count this library will try to allocate.
 *
 * Not a limit the language states: a typed array's length is bounded by what
 * the engine will hand out for the buffer behind it, which is smaller than this
 * on most machines and not written down anywhere. This is a portable ceiling
 * that turns an ask no engine will meet into one error rather than nine.
 */
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
export function allocateScores<K extends ScoreArrayKind>(
  kind: K,
  length: number,
  what: string,
): ScoreArrayOf[K] {
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
  const { range } = scoreArrayFactory(kind)
  if (range === null) return null
  const low = roundHalfAwayFromZero(
    Math.min(bounds[0] * multiplier, bounds[1] * multiplier),
  )
  const high = roundHalfAwayFromZero(
    Math.max(bounds[0] * multiplier, bounds[1] * multiplier),
  )
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

/**
 * A `queries × choices` block of scores, stored row-major in one typed array.
 *
 * Storage backing `scoreMatrix`. Two-dimensional
 * indexing is the only thing a flat array does not already give a caller, so
 * that is all this adds.
 */
export interface ScoreMatrix<A extends ScoreArray = Float64Array> {
  /** Number of queries. */
  readonly rows: number
  /** Number of choices. */
  readonly cols: number
  /** Row-major scores, `rows * cols` long. `data[row * cols + col]`. */
  readonly data: A
  /** The score of `queries[row]` against `choices[col]`. */
  at(row: number, col: number): number
  /** A copy as nested plain arrays. */
  toArray(): number[][]
  /** Each row in turn, as a view over {@link data} rather than a copy. */
  [Symbol.iterator](): IterableIterator<A>
}

/**
 * Check a dimension on its own, because the allocation only ever sees their
 * product: `-1 × -1` is a length of one, and so is `0.5 × 2`. Either would
 * build a matrix whose `at`, `toArray` and row iterator all disagree with the
 * data behind them, and `allocateScores` would accept both.
 *
 * `scoreMatrix` cannot reach this — its dimensions are array lengths — so what
 * it guards is this module's own contract: nothing here should be able to
 * return an internally inconsistent {@link ScoreMatrix}.
 */
function validateDimension(value: number, name: string, what: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${what} needs a ${name} count, not ${value}`)
  }
}

/**
 * Allocate a matrix of `kind`, fill it, and wrap it.
 *
 * `kind` has to arrive as a single literal rather than the union, which is what
 * lets `view` come back at the concrete element type and the row iterator
 * promise `A` instead of the whole union. Callers holding a runtime kind
 * dispatch to a literal first — see `scoreMatrix`.
 */
export function buildScoreMatrix<K extends ScoreArrayKind>(
  kind: K,
  rows: number,
  cols: number,
  what: string,
  fill: (data: ScoreArrayOf[K], integral: boolean) => void,
): ScoreMatrix<ScoreArrayOf[K]> {
  validateDimension(rows, 'row', what)
  validateDimension(cols, 'column', what)
  const { integral, view } = scoreArrayFactory(kind)
  const data = allocateScores(kind, rows * cols, what)
  fill(data, integral)

  return {
    rows,
    cols,
    data,
    at(row, col) {
      // A `RangeError` rather than `undefined`, which is what keeps the return
      // type honestly `number` under `noUncheckedIndexedAccess: false`.
      if (
        !Number.isInteger(row) ||
        row < 0 ||
        row >= rows ||
        !Number.isInteger(col) ||
        col < 0 ||
        col >= cols
      ) {
        throw new RangeError(`(${row}, ${col}) is outside a ${rows} × ${cols} matrix`)
      }
      return data[row * cols + col]
    },
    toArray() {
      const out = new Array<number[]>(rows)
      for (let i = 0; i < rows; i++) {
        const row = new Array<number>(cols)
        const base = i * cols
        for (let j = 0; j < cols; j++) row[j] = data[base + j]
        out[i] = row
      }
      return out
    },
    *[Symbol.iterator]() {
      for (let i = 0; i < rows; i++) yield view(data, i * cols, (i + 1) * cols)
    },
  }
}
