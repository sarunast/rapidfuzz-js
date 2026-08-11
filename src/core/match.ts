import { scorerCompilation, type Scorer, type ThresholdOptions } from './scorer.js'
import { trustedKernelThreshold, validateThreshold } from './threshold.js'
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
  // A threshold no score in the bounds can fail is one the kernel need not be
  // asked about — but the inputs still have to be legal, which is what
  // `validate` settles. Through `trustedKernelThreshold` rather than a test
  // written out here: it is the same question `Scorer.score` asks to decide
  // whether to pass the kernel a cutoff at all, and it answers `null` for a
  // distance at its upper bound as well as a similarity at its lower one. The
  // test written out here covered only the second, so `isMatch` at
  // `normalizedDistance`'s threshold of 1 scored every pair to learn what its
  // bounds already said.
  if (
    compilation.trusted &&
    trustedKernelThreshold(compilation.direction, compilation.bounds, threshold) === null
  ) {
    compilation.validate(a, b)
    return true
  }
  return scorer.score(a, b, { threshold }) !== undefined
}
