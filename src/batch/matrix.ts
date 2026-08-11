import { scorerCompilation, type Scorer } from '../core/scorer.js'
import { validateSequence } from '../core/sequence.js'
import { qualifies } from '../core/threshold.js'
import type { Direction, Normalizer, Sequence } from '../core/types.js'
import { resolveBatchOptions } from './options.js'
import {
  buildScoreMatrix,
  roundHalfAwayFromZero,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
  type ScoreMatrix,
} from './scoreArray.js'
import type { BatchOptions } from './types.js'

function normalizeInputs(
  values: readonly Sequence[],
  normalize: Normalizer | undefined,
): readonly Sequence[] {
  return values.map((value) => {
    const sequence = validateSequence(value)
    if (normalize === undefined) return sequence
    const normalized = normalize(sequence)
    if (normalized == null) throw new TypeError('normalize returned a missing value')
    return validateSequence(normalized)
  })
}

function fill<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  scorer: Scorer<D>,
  store: ScoreArray,
  integral: boolean,
  symmetric: boolean,
  threshold: number | null,
  multiplier: number,
): void {
  const compilation = scorerCompilation(scorer)
  const preparedChoices = choices.map(compilation.prepareChoice)
  const columns = choices.length
  for (let row = 0; row < queries.length; row++) {
    const prepared = compilation.prepareQuery(queries[row])
    const start = symmetric ? row : 0
    for (let column = start; column < columns; column++) {
      const raw = prepared(preparedChoices[column], threshold)
      const score =
        compilation.trusted ||
        threshold === null ||
        qualifies(compilation.direction, raw, threshold)
          ? raw
          : compilation.direction === 'similarity'
            ? compilation.bounds[0]
            : compilation.bounds[1]
      const scaled = score * multiplier
      const stored = integral ? roundHalfAwayFromZero(scaled) : scaled
      store[row * columns + column] = stored
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
  const kind = options.into ?? 'f64'
  const { threshold, multiplier } = resolveBatchOptions(
    options.threshold,
    options.scoreMultiplier,
  )
  const sameInput = Object.is(queries, choices)
  const normalizedChoices = normalizeInputs(choices, options.normalize)
  const normalizedQueries = sameInput
    ? normalizedChoices
    : normalizeInputs(queries, options.normalize)
  const symmetric = sameInput && options.scorer.symmetric
  return buildScoreMatrix(
    kind,
    normalizedQueries.length,
    normalizedChoices.length,
    'scoreMatrix',
    (data, integral) =>
      fill(
        normalizedQueries,
        normalizedChoices,
        options.scorer,
        data,
        integral,
        symmetric,
        threshold,
        multiplier,
      ),
  )
}
