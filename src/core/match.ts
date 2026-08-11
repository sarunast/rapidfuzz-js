import { scorerCompilation, type Scorer, type ThresholdOptions } from './scorer.js'
import { validateThreshold } from './threshold.js'
import type { Direction, MaybeSequence } from './types.js'

export function scoreIfMatch<D extends Direction>(
  scorer: Scorer<D>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): number | undefined {
  return scorer.score(a, b, options)
}

export function isMatch<D extends Direction>(
  scorer: Scorer<D>,
  a: MaybeSequence,
  b: MaybeSequence,
  options: ThresholdOptions,
): boolean {
  const threshold = validateThreshold(options.threshold)
  const compilation = scorerCompilation(scorer)
  if (
    compilation.trusted &&
    ((compilation.direction === 'similarity' && threshold <= compilation.bounds[0]) ||
      (compilation.direction === 'distance' && threshold >= compilation.bounds[1]))
  ) {
    compilation.validate(a, b)
    return true
  }
  return scorer.score(a, b, { threshold }) !== undefined
}
