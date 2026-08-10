import type { MaybeSequence } from '../src/_common.js'
import type { ScoreArrayKind } from '../src/_scoreArray.js'
/**
 * Plain-array views of `scoreMatrix` and `scorePairs`, for the ported tests.
 *
 * Upstream's `cdist` returns a NumPy array and its tests assert against nested
 * Python lists. This port used to return `number[][]`, so those assertions
 * carried over as written; now the result is a {@link ScoreMatrix} over one
 * typed array. Unwrapping it here rather than at ~120 call sites keeps every
 * ported assertion the shape upstream wrote it in, which is the point of
 * porting them.
 *
 * The matrix API itself — `at`, `data`, `into`, row iteration — is exercised in
 * `scoreArray.test.ts` instead, where it is the subject rather than the vehicle.
 */
import { scoreMatrix, scorePairs, type ScoreOptions } from '../src/search.js'

/** `scoreMatrix(...).toArray()`. */
export function matrixScores(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind> = {},
): number[][] {
  return scoreMatrix(queries, choices, options).toArray()
}

/** `Array.from(scorePairs(...))`. */
export function pairScores(
  queries: readonly MaybeSequence[],
  choices: readonly MaybeSequence[],
  options: ScoreOptions<ScoreArrayKind> = {},
): number[] {
  return Array.from(scorePairs(queries, choices, options))
}
