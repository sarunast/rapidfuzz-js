import { scorerCompilation, type Scorer } from '../core/scorer.js'
import { validateSequence } from '../core/sequence.js'
import type { Direction, Sequence } from '../core/types.js'
import {
  buildScoreMatrix,
  roundHalfAwayFromZero,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
  type ScoreMatrix,
} from './scoreArray.js'
import type { BatchOptions } from './types.js'

function fill<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  scorer: Scorer<D>,
  store: ScoreArray,
  integral: boolean,
): void {
  const compilation = scorerCompilation(scorer)
  let offset = 0
  for (const query of queries) {
    const prepared = compilation.prepareQuery(validateSequence(query))
    for (const choice of choices) {
      const score = prepared(compilation.prepareChoice(validateSequence(choice)), null)
      store[offset++] = integral ? roundHalfAwayFromZero(score) : score
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
  return buildScoreMatrix(
    kind,
    queries.length,
    choices.length,
    'scoreMatrix',
    (data, integral) => fill(queries, choices, options.scorer, data, integral),
  )
}
