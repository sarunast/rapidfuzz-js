import { validateThreshold } from '../core/threshold.js'
import type { Direction } from '../core/types.js'

export interface ResolvedBatchOptions {
  readonly threshold: number | null
  readonly multiplier: number
}

export function resolveBatchOptions(
  threshold: number | undefined,
  multiplier: number | undefined,
): ResolvedBatchOptions {
  const resolvedMultiplier = multiplier ?? 1
  if (!Number.isFinite(resolvedMultiplier)) {
    throw new RangeError('scoreMultiplier must be finite')
  }
  return {
    threshold: threshold === undefined ? null : validateThreshold(threshold),
    multiplier: resolvedMultiplier,
  }
}

/**
 * The **natural-scale** score a rejected pair takes, refused when the value it
 * would be stored as cannot represent a rejection.
 *
 * The scale matters: the caller multiplies and rounds this afterwards, exactly
 * as it does a real score, so returning an already-scaled value would apply
 * `scoreMultiplier` twice. `multiplier` and `integral` are taken only to decide
 * whether that later arithmetic can represent the answer.
 *
 * Only an untrusted scorer reaches this: a built-in expresses its own rejection
 * through the cutoff it was given. A custom scorer expresses it through the
 * far end of its declared bounds, and those may legitimately be `[0, Infinity]`
 * for a distance — which is a bound, not a score. Written into an integer
 * destination `Infinity` becomes `0`, the *best* distance there is, and a
 * `scoreMultiplier` of `0` turns it into `NaN`. Both read as an answer, so the
 * call is refused once, up front, rather than filling an array with one.
 */
export function rejectedScore(
  direction: Direction,
  bounds: readonly [number, number],
  multiplier: number,
  integral: boolean,
): number {
  const rejected = direction === 'similarity' ? bounds[0] : bounds[1]
  const stored = rejected * multiplier
  if (Number.isNaN(stored) || (integral && !Number.isFinite(stored))) {
    throw new RangeError(
      'a thresholded batch cannot store this scorer’s rejected score: give the ' +
        'scorer finite bounds, drop the threshold, or score into f64',
    )
  }
  return rejected
}
