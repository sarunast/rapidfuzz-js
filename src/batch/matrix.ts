import { assertOptionKeys } from '../core/options.js'
import type { MetricCompilation } from '../core/protocol.js'
import { scorerCompilation } from '../core/scorer.js'
import { normalizeSequence, validateSequence } from '../core/sequence.js'
import { qualifies } from '../core/threshold.js'
import type { Direction, Normalizer, Sequence } from '../core/types.js'
import { BATCH_OPTION_KEYS, rejectedScore, resolveBatchOptions } from './options.js'
import {
  buildScoreMatrix,
  roundHalfAwayFromZero,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
  type ScoreMatrix,
  scoreStoreRange,
  unstorableScore,
} from './scoreArray.js'
import type { BatchOptions } from './types.js'

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

export function scoreMatrix<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, 'f64'>,
): ScoreMatrix<Float64Array>
export function scoreMatrix<D extends Direction, K extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, K> & { readonly into: K },
): ScoreMatrix<ScoreArrayOf[K]>
export function scoreMatrix<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, ScoreArrayKind>,
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
