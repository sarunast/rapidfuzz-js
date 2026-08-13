import type { MetricCompilation } from '../core/protocol.js'
import { impossibleTrustedThreshold, optionalThreshold } from '../core/threshold.js'
import type { Direction } from '../core/types.js'
import type { ScoreArrayKind } from './scoreArray.js'
import type { BatchOptions } from './types.js'

export const BATCH_OPTION_KEYS: readonly string[] = [
  'scorer',
  'into',
  'normalize',
  'threshold',
  'scoreMultiplier',
] as const satisfies readonly (keyof BatchOptions<Direction, ScoreArrayKind>)[]

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
    threshold: optionalThreshold(threshold),
    multiplier: resolvedMultiplier,
  }
}

/**
 * The **natural-scale** score a rejected pair takes, or `null` when the kernel
 * states its own rejection and the caller's threshold need not be re-tested.
 *
 * A trusted kernel given a cutoff returns a value outside it — `cutoff + 1` for
 * a raw distance, `0` for a similarity — which is what `process.cdist` stores
 * upstream. Two cases have no such sentinel: a custom metric, which never sees
 * the cutoff, and a threshold outside the scorer's bounds, where `cutoff + 1`
 * is a real score — an impossible `-1` on a distance yields a perfect `0`.
 * Both fall back to the far end of the declared bounds.
 *
 * The scale matters: the caller multiplies and rounds this afterwards, exactly
 * as it does a real score, so returning an already-scaled value would apply
 * `scoreMultiplier` twice. `multiplier` and `integral` are taken only to decide
 * whether that later arithmetic can represent the answer. The far bound may
 * legitimately be `Infinity` for a distance — which is a bound, not a score.
 * Written into an integer destination it becomes `0`, the *best* distance there
 * is, and a `scoreMultiplier` of `0` turns it into `NaN`. Both read as an
 * answer, so the call is refused up front rather than filling an array with one.
 */
export function rejectedScore(
  compilation: MetricCompilation<Direction>,
  threshold: number | null,
  multiplier: number,
  integral: boolean,
): number | null {
  const { direction, bounds } = compilation
  if (
    threshold === null ||
    (compilation.trusted && !impossibleTrustedThreshold(direction, bounds, threshold))
  ) {
    return null
  }
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
