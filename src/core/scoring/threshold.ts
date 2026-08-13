import type { Direction } from '../types.js'
import type { MetricCompilation } from './compilation.js'

/**
 * Finite even for a scorer whose upper bound is `Infinity`: "everything
 * qualifies" is what `null` says internally, and one value with two meanings on
 * two sides of a boundary is how the two stop agreeing.
 */
export function validateThreshold(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

/** An absent threshold as the `null` every internal caller reads. */
export function optionalThreshold(value: number | undefined): number | null {
  return value === undefined ? null : validateThreshold(value)
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
 * {@link qualifies} where the caller may not have asked for a threshold at all,
 * which every score passes.
 *
 * Written out rather than delegating: both forms sit in scan loops that run once
 * per candidate, and the pair is two comparisons either way.
 */
export function passesThreshold(
  direction: Direction,
  score: number,
  threshold: number | null,
): boolean {
  return (
    threshold === null ||
    (direction === 'similarity' ? score >= threshold : score <= threshold)
  )
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

/**
 * The three questions a search asks of a threshold before it reaches a kernel,
 * each answered once for both kinds of compilation.
 *
 * A custom metric's bounds are the caller's claim rather than the algorithm's,
 * so nothing may be concluded from them: no threshold is impossible, the kernel
 * is told exactly what the caller asked for, and no score is known to be
 * unbeatable. Spelling that out at every call site is how one of them ends up
 * pruning a scorer that never agreed to be pruned.
 */
export function impossibleThreshold(
  compilation: MetricCompilation<Direction>,
  threshold: number | null,
): boolean {
  return (
    compilation.trusted &&
    impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
  )
}

/** What the kernel is told, given {@link impossibleThreshold} answered `false`. */
export function kernelThreshold(
  compilation: MetricCompilation<Direction>,
  threshold: number | null,
): number | null {
  return compilation.trusted
    ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
    : threshold
}

/**
 * The score nothing can beat, or `null` when there is no such score to stop a
 * scan on.
 */
export function knownOptimum(compilation: MetricCompilation<Direction>): number | null {
  if (!compilation.trusted) return null
  return compilation.direction === 'similarity'
    ? compilation.bounds[1]
    : compilation.bounds[0]
}
