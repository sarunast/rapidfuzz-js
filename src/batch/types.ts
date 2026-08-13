import type { Scorer } from '../core/scoring/scorer.js'
import type { Direction, Normalizer } from '../core/types.js'
import type { ScoreArrayKind } from './scoreArray.js'

/** Shared by {@link scoreMatrix} and {@link scorePairs}. */
export interface BatchOptions<
  TDirection extends Direction,
  TKind extends ScoreArrayKind = 'f64',
> {
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
