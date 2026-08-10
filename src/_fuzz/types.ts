/**
 * The data contracts the fuzz scorers share, and nothing else.
 *
 * A dependency-free leaf: every other module under `_fuzz/` may import this one,
 * and this one imports only types from `_common.ts`. Keeping it free of runtime
 * logic is what lets the scorer families below it stay unaware of each other.
 */
import type { MaybeSequence, Processor } from '../_common.js'

/**
 * Options every fuzz scorer accepts.
 *
 * Spelled out rather than extending `ScorerOptions`, although the three fields
 * are the same, because `scoreCutoff` means something different here: these
 * scorers report a percentage, so its range is `[0, 100]` and the docs a reader
 * gets from this interface should say so.
 */
export interface FuzzOptions {
  /** Applied to both inputs before scoring. */
  readonly processor?: Processor | undefined
  /** A threshold in `[0, 100]`. Scores below it are returned as `0`. */
  readonly scoreCutoff?: number | undefined
  /**
   * Performance hint only; it cannot change the score.
   *
   * Accepted and currently ignored by every fuzz scorer, which is a measured
   * position rather than an unfinished one. These scorers reach the
   * bit-parallel LCS kernel, whose only budgeted lever is an early exit — and
   * an early exit sized by an *estimate* has to be redone whenever the estimate
   * was too optimistic. Wiring it into prepared `ratio` cost 1.49-1.71x the
   * kernel iterations across an `extract` of 2000 choices, so it was dropped.
   * `search`'s `extractOne` already tightens `scoreCutoff` to the best score it
   * has seen, which is the same pruning exactly rather than approximately.
   *
   * The distance scorers are the ones that honour it — `levenshtein` sizes its
   * band from the hint, where a wrong guess widens rather than restarts.
   */
  readonly scoreHint?: number | undefined
}

/**
 * Anything a fuzz scorer accepts, including the "missing value" cases.
 *
 * An alias rather than a restatement: it is the same set of inputs the
 * normalized scorers in `distance/` take, and the reason `NaN` is missing from
 * it — treated as missing at run time, kept out of the type — is written up
 * there, on {@link MaybeSequence}.
 */
export type FuzzInput = MaybeSequence

/**
 * Where the best alignment of the shorter input sits inside the longer one.
 *
 * A result, so read-only: nothing reads these fields back after they are
 * returned. The search that produces one works on a plain mutable record and
 * lets it widen to this on the way out.
 */
export interface ScoreAlignment {
  readonly score: number
  readonly srcStart: number
  readonly srcEnd: number
  readonly destStart: number
  readonly destEnd: number
}

/** Which scorer a prepared-query hook was built for. See `prepared.ts`. */
export type PreparedFuzzKind =
  | 'ratio'
  | 'partialRatio'
  | 'tokenSortRatio'
  | 'tokenSetRatio'
  | 'tokenRatio'
  | 'partialTokenSortRatio'
  | 'partialTokenSetRatio'
  | 'partialTokenRatio'
  | 'wRatio'
  | 'qRatio'
