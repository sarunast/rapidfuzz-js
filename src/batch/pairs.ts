import { assertOptionKeys } from '../core/options.js'
import { scorerCompilation } from '../core/scorer.js'
import { normalizeSequence, validateSequence } from '../core/sequence.js'
import { qualifies } from '../core/threshold.js'
import type { Direction, Normalizer, Sequence } from '../core/types.js'
import { BATCH_OPTION_KEYS, rejectedScore, resolveBatchOptions } from './options.js'
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
  return values.map((value) => normalizeSequence(validateSequence(value), normalize))
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
  assertOptionKeys(options, BATCH_OPTION_KEYS, 'scorePairs')
  if (queries.length !== choices.length) {
    throw new RangeError('queries and choices must have the same length')
  }
  // Configuration first, allocation second, data third. Every check below is a
  // few comparisons, and reaching them after the allocation meant a bad
  // threshold or an unusable multiplier was reported only once a typed array
  // the length of the input had been handed out.
  const kind = options.into ?? 'f64'
  const compilation = scorerCompilation(options.scorer)
  const { threshold, multiplier } = resolveBatchOptions(
    options.threshold,
    options.scoreMultiplier,
  )
  const scores = allocateScores(kind, queries.length, 'scorePairs')
  const integral = kind !== 'f64' && kind !== 'f32'
  const rejected = rejectedScore(compilation, threshold, multiplier, integral)
  // One closure, with the invariant tests inside it. Choosing between two
  // closures — one that qualifies and one that cannot — measured 1.02-1.18x
  // *slower* over six pair workloads, worst on the custom-scorer case it was
  // meant to help: two shapes reaching one call site is what stops the call
  // being inlined, and that costs more than the branches it removes.
  const store = (score: number): number => {
    const thresholded =
      rejected === null ||
      threshold === null ||
      qualifies(compilation.direction, score, threshold)
        ? score
        : rejected
    const scaled = thresholded * multiplier
    return integral ? roundHalfAwayFromZero(scaled) : scaled
  }
  const sameInput = Object.is(queries, choices)
  if (options.normalize === undefined) {
    for (let i = 0; i < queries.length; i++) {
      const query = validateSequence(queries[i])
      const choice = sameInput ? query : validateSequence(choices[i])
      scores[i] = store(compilation.rawScore(query, choice, threshold))
    }
    return scores
  }
  const normalizedQueries = normalizeInputs(queries, options.normalize)
  const normalizedChoices = sameInput
    ? normalizedQueries
    : normalizeInputs(choices, options.normalize)
  for (let i = 0; i < normalizedQueries.length; i++) {
    scores[i] = store(
      compilation.rawScore(normalizedQueries[i], normalizedChoices[i], threshold),
    )
  }
  return scores
}
