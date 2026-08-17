import type { AnyMetricCompilation } from '#core/scoring/compilation.js'
import type { Scorer } from '#core/scoring/scorer.js'
import { impossibleTrustedThreshold, optionalThreshold } from '#core/scoring/threshold.js'
import type { Direction, Normalizer } from '#core/types.js'

import type { ScoreArrayKind } from './storage.js'

/** Shared by `scoreMatrix` and `scorePairs`. */
export interface BatchOptions<
  TDirection extends Direction,
  TKind extends ScoreArrayKind = 'f64',
> {
  /**
   * What every pair is measured with, from `createScorer`. Its direction and
   * its scale are what {@link threshold} is read against, and what a rejected
   * pair is stored as.
   */
  readonly scorer: Scorer<TDirection>

  /** Element type the scores are stored as. Defaults to `'f64'`. */
  readonly into?: TKind | undefined

  /** Applied to every query and every choice before scoring. */
  readonly normalize?: Normalizer | undefined

  /**
   * The score a pair has to reach, on the scorer's own scale — `0..100` for a
   * fuzz scorer, `0..1` for a normalized one, a count for a raw edit distance.
   *
   * A pair that misses it is stored as its scorer's own rejection, which is
   * what `process.cdist` stores upstream: `threshold + 1` for a raw distance,
   * `0` for a similarity. A custom metric never sees the threshold, so a pair
   * it rejects is stored as the far end of its declared bounds instead — as is
   * every pair under a threshold outside those bounds, which nothing can meet.
   *
   * Read before {@link scoreMultiplier}, so the threshold means the same thing
   * whatever the multiplier is.
   */
  readonly threshold?: number | undefined

  /**
   * Multiplies every score after thresholding and before storage. Defaults to
   * `1`. May be negative — `-1` turns a similarity into a rank key that sorts
   * ascending — and an integral `into` rounds the product half away from zero.
   */
  readonly scoreMultiplier?: number | undefined
}

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
  compilation: AnyMetricCompilation<Direction>,
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
