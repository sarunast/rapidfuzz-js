import { scorerCompilation } from '../core/scorer.js'
import { validateSequence } from '../core/sequence.js'
import type { Direction, Normalizer, Sequence } from '../core/types.js'
import {
  allocateScores,
  roundHalfAwayFromZero,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
} from './scoreArray.js'
import type { BatchOptions } from './types.js'

function normalizeInputs(
  values: readonly Sequence[],
  normalize: Normalizer,
): readonly Sequence[] {
  return values.map((value) => {
    const sequence = validateSequence(value)
    const normalized = normalize(sequence)
    if (normalized == null) throw new TypeError('normalize returned a missing value')
    return validateSequence(normalized)
  })
}

export function scorePairs<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, 'f64'>,
): Float64Array
export function scorePairs<D extends Direction, K extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, K> & { readonly into: K },
): ScoreArrayOf[K]
export function scorePairs<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<D, ScoreArrayKind>,
): ScoreArray {
  if (queries.length !== choices.length) {
    throw new RangeError('queries and choices must have the same length')
  }
  const kind = options.into ?? 'f64'
  const scores = allocateScores(kind, queries.length, 'scorePairs')
  const integral = kind !== 'f64' && kind !== 'f32'
  const compilation = scorerCompilation(options.scorer)
  const sameInput = Object.is(queries, choices)
  if (options.normalize === undefined) {
    for (let i = 0; i < queries.length; i++) {
      const query = validateSequence(queries[i])
      const choice = sameInput ? query : validateSequence(choices[i])
      const score = compilation.rawScore(query, choice, null)
      scores[i] = integral ? roundHalfAwayFromZero(score) : score
    }
    return scores
  }
  const normalizedQueries = normalizeInputs(queries, options.normalize)
  const normalizedChoices = sameInput
    ? normalizedQueries
    : normalizeInputs(choices, options.normalize)
  for (let i = 0; i < normalizedQueries.length; i++) {
    const score = compilation.rawScore(normalizedQueries[i], normalizedChoices[i], null)
    scores[i] = integral ? roundHalfAwayFromZero(score) : score
  }
  return scores
}
