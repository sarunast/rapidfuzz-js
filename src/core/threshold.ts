import type { Direction } from './types.js'

/**
 * Finite even for a scorer whose upper bound is `Infinity`: "everything
 * qualifies" is what `null` says internally, and one value with two meanings on
 * two sides of a boundary is how the two stop agreeing.
 */
export function validateThreshold(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

/** Always applied to the caller's threshold, never the kernel's. */
export function qualifies(
  direction: Direction,
  score: number,
  threshold: number,
): boolean {
  return direction === 'similarity' ? score >= threshold : score <= threshold
}

/**
 * Outside the bounds, so nothing can satisfy it and the algorithm need not run.
 * Strict: a threshold *at* the bound is what an identical pair scores. `null`
 * is "no threshold asked for", which nothing can fail.
 */
export function impossibleTrustedThreshold(
  direction: Direction,
  bounds: readonly [number, number],
  threshold: number | null,
): boolean {
  return (
    threshold !== null &&
    (direction === 'similarity' ? threshold > bounds[1] : threshold < bounds[0])
  )
}

/**
 * The cutoff the kernel is given, `null` when the threshold cannot reject
 * anything within the bounds and pruning would be wasted. Inclusive here where
 * the test above is strict: every score meets a threshold at the bound.
 *
 * Precondition: impossible thresholds are settled first. This returns one
 * unchanged rather than answering for it.
 */
export function trustedKernelThreshold(
  direction: Direction,
  bounds: readonly [number, number],
  threshold: number | null,
): number | null {
  if (threshold === null) return null
  return direction === 'similarity'
    ? threshold <= bounds[0]
      ? null
      : threshold
    : threshold >= bounds[1]
      ? null
      : threshold
}
