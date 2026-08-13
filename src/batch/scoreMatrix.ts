import { assertOptionKeys } from '../core/options.js'
import type { MetricCompilation } from '../core/scoring/compilation.js'
import { scorerCompilation } from '../core/scoring/scorer.js'
import { qualifies } from '../core/scoring/threshold.js'
import { normalizeSequence, validateSequence } from '../core/sequence.js'
import type { Direction, Normalizer, Sequence } from '../core/types.js'
import {
  BATCH_OPTION_KEYS,
  type BatchOptions,
  rejectedScore,
  resolveBatchOptions,
} from './options.js'
import {
  allocateScores,
  roundHalfAwayFromZero,
  type ScoreArray,
  scoreArrayFactory,
  type ScoreArrayKind,
  type ScoreArrayOf,
  scoreStoreRange,
  unstorableScore,
} from './storage.js'

/**
 * A `queries × choices` block of scores, stored row-major in one typed array.
 *
 * Storage backing {@link scoreMatrix}. Two-dimensional
 * indexing is the only thing a flat array does not already give a caller, so
 * that is all this adds.
 */
export interface ScoreMatrix<TArray extends ScoreArray = Float64Array> {
  /** Number of queries. */
  readonly rows: number
  /** Number of choices. */
  readonly cols: number
  /** Row-major scores, `rows * cols` long. `data[row * cols + col]`. */
  readonly data: TArray
  /** The score of `queries[row]` against `choices[col]`. */
  at(row: number, col: number): number
  /** A copy as nested plain arrays. */
  toArray(): number[][]
  /** Each row in turn, as a view over {@link data} rather than a copy. */
  [Symbol.iterator](): IterableIterator<TArray>
}

// The normalizer is fixed for the whole call, so it decides which loop runs
// rather than being re-tested per sequence — the same split `scorePairs` makes.
function normalizeInputs(
  values: readonly Sequence[],
  normalize: Normalizer | undefined,
): readonly Sequence[] {
  if (normalize === undefined) return values.map((value) => validateSequence(value))
  return values.map((value) => normalizeSequence(validateSequence(value), normalize))
}

function fill(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  compilation: MetricCompilation<Direction>,
  kind: ScoreArrayKind,
  store: ScoreArray,
  integral: boolean,
  symmetric: boolean,
  threshold: number | null,
  multiplier: number,
): void {
  const rejected = rejectedScore(compilation, threshold, multiplier, integral)
  // Read out of the tuple once: what the cell loop below can afford is a
  // predictable test against three loop-invariant locals, which is what a
  // scorer whose whole scaled range is provably storable then skips entirely.
  const limit = scoreStoreRange(kind, compilation.bounds, multiplier)
  const bounded = limit !== null
  const lowest = limit === null ? 0 : limit[0]
  const highest = limit === null ? 0 : limit[1]
  // After the rejection check, so an unusable one is still reported, and before
  // preparing the choices: with no rows there is no cell to score them for.
  if (queries.length === 0) return
  const columns = choices.length
  // Written out rather than `choices.map(compilation.prepareChoice)`: the
  // protocol's preparer takes one argument, and `map` would hand it three.
  const preparedChoices = new Array<unknown>(columns)
  for (let column = 0; column < columns; column++) {
    preparedChoices[column] = compilation.prepareChoice(choices[column])
  }
  // The cell loop keeps the invariant tests inline. Splitting it into a trusted
  // loop with no `qualifies` call and a custom loop that post-filters measured
  // 0.99-1.00x on five built-in matrices — including a 50x200 `similarity`
  // matrix at 55ns a cell, where the plumbing has its largest possible share —
  // and 1.02x on the custom scorer it was meant to leave alone. Two loop bodies
  // for a number this machine cannot resolve is not a trade worth making.
  for (let row = 0; row < queries.length; row++) {
    const prepared = compilation.prepareQuery(queries[row])
    const rowOffset = row * columns
    const start = symmetric ? row : 0
    for (let column = start; column < columns; column++) {
      const raw = prepared(preparedChoices[column], threshold)
      const score =
        rejected === null ||
        threshold === null ||
        qualifies(compilation.direction, raw, threshold)
          ? raw
          : rejected
      const scaled = score * multiplier
      const stored = integral ? roundHalfAwayFromZero(scaled) : scaled
      // Negated, so a `NaN` could not pass the way it passes a comparison.
      if (bounded && !(stored >= lowest && stored <= highest)) {
        unstorableScore(stored, kind, 'scoreMatrix')
      }
      store[rowOffset + column] = stored
      if (symmetric && row !== column) store[column * columns + row] = stored
    }
  }
}

/**
 * Check a dimension on its own, because the allocation only ever sees their
 * product: `-1 × -1` is a length of one, and so is `0.5 × 2`. Either would
 * build a matrix whose `at`, `toArray` and row iterator all disagree with the
 * data behind them, and `allocateScores` would accept both.
 *
 * {@link scoreMatrix} cannot reach this — its dimensions are array lengths — so
 * what it guards is {@link buildScoreMatrix}'s own contract: nothing there
 * should be able to return an internally inconsistent {@link ScoreMatrix}.
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
 * promise `TArray` instead of the whole union. Callers holding a runtime kind
 * dispatch to a literal first — see {@link scoreMatrix}.
 */
export function buildScoreMatrix<TKind extends ScoreArrayKind>(
  kind: TKind,
  rows: number,
  cols: number,
  what: string,
  fill: (data: ScoreArrayOf[TKind], integral: boolean) => void,
): ScoreMatrix<ScoreArrayOf[TKind]> {
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

/**
 * Score every query against every choice, into one typed array.
 *
 * ```ts
 * const matrix = scoreMatrix(['cat', 'dog'], ['cats', 'dogs'], { scorer })
 * matrix.at(0, 1) // score('cat', 'dogs')
 * matrix.data // row-major Float64Array, rows * cols long
 * ```
 *
 * An `R × C` matrix costs `R + C` preparations rather than `R × C`: each query
 * is prepared once for its row and each choice once for its column. Scores are
 * written straight into the store, so no result object is allocated per pair.
 *
 * A symmetric scorer over identical inputs is scored once per unordered pair
 * and mirrored.
 *
 * Use this when you want the whole grid of numbers. When the question is
 * "which choices match best", that is a search — `bestMatch` or `search` —
 * which can prune with a threshold instead of scoring everything.
 *
 * @param queries Rows of the matrix.
 * @param choices Columns of the matrix.
 * @returns A {@link ScoreMatrix} view over the store: `at(row, col)`,
 * `toArray()`, and row-wise iteration, all over `data` without copying.
 * @throws `TypeError` for an unknown option key, or for an operand that is not
 * a valid sequence.
 * @throws `RangeError` if a score does not fit the `into` element type.
 */
export function scoreMatrix<TDirection extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, 'f64'>,
): ScoreMatrix<Float64Array>
export function scoreMatrix<TDirection extends Direction, TKind extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, TKind> & { readonly into: TKind },
): ScoreMatrix<ScoreArrayOf[TKind]>
export function scoreMatrix<TDirection extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, ScoreArrayKind>,
): ScoreMatrix<ScoreArray> {
  assertOptionKeys(options, BATCH_OPTION_KEYS, 'scoreMatrix')
  // Configuration first, data second — the order `scorePairs` uses. Reaching
  // the scorer only inside `fill` meant a `normalize` with a side effect ran,
  // and a whole matrix was allocated, before a scorer this package did not
  // build was refused.
  const kind = options.into ?? 'f64'
  const compilation = scorerCompilation(options.scorer)
  const { threshold, multiplier } = resolveBatchOptions(
    options.threshold,
    options.scoreMultiplier,
  )
  const sameInput = Object.is(queries, choices)
  const normalizedChoices = normalizeInputs(choices, options.normalize)
  const normalizedQueries = sameInput
    ? normalizedChoices
    : normalizeInputs(queries, options.normalize)
  const symmetric = sameInput && compilation.symmetric
  return buildScoreMatrix(
    kind,
    normalizedQueries.length,
    normalizedChoices.length,
    'scoreMatrix',
    (data, integral) =>
      fill(
        normalizedQueries,
        normalizedChoices,
        compilation,
        kind,
        data,
        integral,
        symmetric,
        threshold,
        multiplier,
      ),
  )
}
