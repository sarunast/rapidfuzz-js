import { scorerCompilation, type Scorer, type ThresholdOptions } from './scorer.js'
import { kernelThreshold, validateThreshold } from './threshold.js'
import type { Direction, MaybeSequence } from './types.js'

/**
 * The score for a pair, or `undefined` when it does not clear the threshold —
 * `scorer.score(a, b, options)` under a name that reads as the question.
 *
 * ```ts
 * scoreIfMatch(scorer, 'kitten', 'sitting', { threshold: 5 }) // 5
 * scoreIfMatch(scorer, 'kitten', 'sitting', { threshold: 3 }) // undefined
 * ```
 *
 * Reach for it when you want the number *and* the verdict; use {@link isMatch}
 * when the number is not going to be used.
 *
 * @param scorer Decides direction, scale, and what a missing operand means.
 * @returns The score when it qualifies, otherwise `undefined` — never a
 * sentinel, because `0` is a legitimate score.
 * @throws `TypeError` if an operand is not a valid sequence, or is missing
 * where the scorer refuses it.
 * @throws `RangeError` if `threshold` is not a finite number.
 */
export function scoreIfMatch<TDirection extends Direction>(
  scorer: Scorer<TDirection>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): number | undefined {
  return scorer.score(a, b, options)
}

/**
 * Whether a pair clears the threshold.
 *
 * ```ts
 * isMatch(scorer, 'kitten', 'sitting', { threshold: 3 }) // false
 * ```
 *
 * Cheaper than comparing a score by hand: the threshold reaches the kernel as a
 * cutoff, so a pair that cannot qualify is abandoned mid-computation, and a
 * threshold no score could fail skips the comparison entirely — while still
 * validating both operands, so a bad input is refused either way.
 *
 * @param scorer Decides direction, scale, and what a missing operand means. For
 * a similarity the threshold is a minimum, for a distance a maximum.
 * @returns Whether the pair qualifies.
 * @throws `TypeError` if an operand is not a valid sequence, or is missing
 * where the scorer refuses it.
 * @throws `RangeError` if `threshold` is not a finite number.
 */
export function isMatch<TDirection extends Direction>(
  scorer: Scorer<TDirection>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): boolean {
  const threshold = validateThreshold(options.threshold)
  const compilation = scorerCompilation(scorer)
  // A threshold no score in the bounds can fail is one the kernel need not be
  // asked about — but the inputs still have to be legal, which is what
  // `validate` settles. Through `kernelThreshold` rather than a test written
  // out here: it is the same question `Scorer.score` asks to decide whether to
  // pass the kernel a cutoff at all, and it answers `null` for a distance at
  // its upper bound as well as a similarity at its lower one. The test written
  // out here covered only the second, so `isMatch` at `normalizedDistance`'s
  // threshold of 1 scored every pair to learn what its bounds already said.
  // `trusted` gates that conclusion and is also what reveals `validate`.
  if (compilation.trusted && kernelThreshold(compilation, threshold) === null) {
    compilation.validate(a, b)
    return true
  }
  return scorer.score(a, b, { threshold }) !== undefined
}
