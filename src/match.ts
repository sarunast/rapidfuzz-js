/**
 * Threshold helpers that answer "is this a match?" without the sentinel.
 *
 * Every scorer bounds its result with `scoreCutoff`, and reports a miss by
 * returning a value rather than by saying so: a distance comes back as
 * `scoreCutoff + 1`, a similarity as `0`, a normalized distance as `1`. That is
 * upstream's protocol and stays exactly as it is — but `0` is also a perfectly
 * real similarity, so a caller reading the number alone cannot always tell a
 * miss from a genuine score.
 *
 * These two express the question directly. Neither has a Python counterpart.
 */
import { callScorer, scorerFlagsOf, toRecord } from './_common.js'

/**
 * A scorer's own options, with `threshold` in place of `scoreCutoff`.
 *
 * Both names are taken out of `O` rather than only `scoreCutoff`: `threshold`
 * *is* the cutoff under a different name, so accepting both could only mean two
 * answers to one question, and a scorer with a `threshold` option of its own
 * cannot reach it through here either way.
 *
 * `object` rather than `ScorerOptions`, because nothing in this module needs a
 * scorer's options to be a scorer's options — `threshold` is passed through as
 * `scoreCutoff` and the answer is decided by comparing the score afterwards,
 * which is exact whether or not the scorer honoured the cutoff at all. The
 * tighter constraint refused the third-party scorers the flags fallback below
 * exists to support: `ScorerOptions` is all-optional, so an unrelated options
 * type shares no property with it and is rejected as a weak type.
 */
export type MatchOptions<O extends object> = Omit<O, 'scoreCutoff' | 'threshold'> & {
  /**
   * The score a match must reach. Read as a minimum for a similarity and as a
   * maximum for a distance, according to the scorer's own flags — so
   * `{ threshold: 80 }` on `ratio` and `{ threshold: 3 }` on
   * `levenshteinDistance` both mean what they look like.
   */
  readonly threshold: number
}

/**
 * The score, or `undefined` when it does not meet `threshold`.
 *
 * ```ts
 * matchScore(ratio, a, b, { threshold: 80 })               // number | undefined
 * matchScore(levenshteinDistance, a, b, { threshold: 3 })  // number | undefined
 * ```
 *
 * Note what this does *not* have to do: decode a sentinel. `threshold` is
 * passed as the scorer's own `scoreCutoff`, and the cutoff contract already
 * guarantees that a miss comes back on the far side of it — below for a
 * similarity, above for a distance. So re-applying the same comparison is
 * exact, including where a sentinel and a real score coincide: `threshold: 0`
 * on a fuzz scorer accepts a genuine zero, and `threshold: 1` on a normalized
 * distance accepts a genuine one, because at those thresholds everything
 * matches and the answer is the score either way.
 *
 * A scorer with no flags is read as a percentage, which is what
 * `scorerFlagsOf` assumes for anything this package did not build.
 */
export function matchScore<I, O extends object>(
  scorer: (s1: I, s2: I, options?: O) => number,
  s1: I,
  s2: I,
  options: MatchOptions<O>,
): number | undefined {
  // `threshold` is destructured out rather than forwarded: it is this module's
  // spelling of the cutoff, not an option any scorer knows.
  const { threshold, ...forScorer } = options
  const { worstScore, optimalScore } = scorerFlagsOf(scorer)

  const score = callScorer(scorer, s1, s2, {
    ...toRecord(forScorer),
    scoreCutoff: threshold,
  })

  return (optimalScore > worstScore ? score >= threshold : score <= threshold)
    ? score
    : undefined
}

/** Whether `s1` and `s2` meet `threshold`. See {@link matchScore}. */
export function isMatch<I, O extends object>(
  scorer: (s1: I, s2: I, options?: O) => number,
  s1: I,
  s2: I,
  options: MatchOptions<O>,
): boolean {
  return matchScore(scorer, s1, s2, options) !== undefined
}
