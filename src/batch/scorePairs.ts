import { assertOptionKeys } from '../core/options.js'
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
  type ScoreArrayKind,
  type ScoreArrayOf,
  scoreStoreRange,
  unstorableScore,
} from './storage.js'

function normalizeInputs(
  values: readonly Sequence[],
  normalize: Normalizer,
): readonly Sequence[] {
  return values.map((value) => normalizeSequence(validateSequence(value), normalize))
}

/**
 * Score two sequences element-wise — `queries[i]` against `choices[i]`.
 *
 * ```ts
 * scorePairs(['cat', 'dog'], ['cats', 'dogs'], { scorer })
 * // Float64Array [85.71…, 85.71…]
 * ```
 *
 * The element-wise counterpart to `scoreMatrix`: use this when the two
 * arrays are already paired up, and the matrix when you need the cross product.
 * Both write into one typed array with no per-score allocation.
 *
 * @param queries One side of each pair.
 * @param choices The other side, positionally matched.
 * @returns One score per index, in the element type `into` names.
 * @throws `TypeError` for an unknown option key, or for an operand that is not
 * a valid sequence.
 * @throws `RangeError` if the two arrays are different lengths, or if a score
 * does not fit the `into` element type.
 */
export function scorePairs<TDirection extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, 'f64'>,
): Float64Array
export function scorePairs<TDirection extends Direction, TKind extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, TKind> & { readonly into: TKind },
): ScoreArrayOf[TKind]
export function scorePairs<TDirection extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: BatchOptions<TDirection, ScoreArrayKind>,
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
  // See `scoreStoreRange`: proven-storable scorers pay nothing, and everything
  // else is tested against three locals the closure closes over.
  const limit = scoreStoreRange(kind, compilation.bounds, multiplier)
  const bounded = limit !== null
  const lowest = limit === null ? 0 : limit[0]
  const highest = limit === null ? 0 : limit[1]
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
    const stored = integral ? roundHalfAwayFromZero(scaled) : scaled
    // Negated, so a `NaN` could not pass the way it passes a comparison.
    if (bounded && !(stored >= lowest && stored <= highest)) {
      unstorableScore(stored, kind, 'scorePairs')
    }
    return stored
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
