/**
 * `wRatio` and `qRatio` — the two scorers that pick a strategy rather than
 * define one.
 *
 * Neither adds an algorithm. `wRatio` chooses among `ratio`, `tokenRatio`,
 * `partialRatio` and `partialTokenRatio` from the length ratio of its two
 * inputs and scales what it gets by 0.95, 0.9 or 0.6; `qRatio` is `ratio` with
 * one different answer for the empty pair.
 *
 * This is the top *raw-input* layer of the scorer graph: it imports `basic`,
 * `tokens` and `tokenScorers`, and only the facade imports it. `prepared.ts`
 * does **not** — it mirrors these two strategies independently over prepared
 * state, because everything an `*_impl` here does first, validating and
 * converting raw input, is already done by the time a prepared branch runs. The
 * two copies have to be kept in step by hand.
 */
import { asSequence, convSequence, isNone } from '../_common.js'
import {
  applyProcessor,
  partialRatioConverted,
  ratioConverted,
  ratio_impl,
} from './basic.js'
import { containsWhitespace } from './tokens.js'
import { partialTokenRatioConverted, tokenRatioConverted } from './tokenScorers.js'
import type { FuzzInput, FuzzOptions } from './types.js'

/**
 * Weighted combination of the other scorers, picking a strategy from the length
 * ratio of the two inputs. Partial matches are scaled by 0.9, misordered full
 * matches by 0.95.
 */
export function wRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const UNBASE_SCALE = 0.95

  const p1 = applyProcessor(asSequence(s1), options.processor)
  const p2 = applyProcessor(asSequence(s2), options.processor)

  // FuzzyWuzzy returns 0 here; kept for compatibility. See RapidFuzz issue 110.
  if (p1.length === 0 || p2.length === 0) return 0

  let scoreCutoff = options.scoreCutoff ?? 0
  if (scoreCutoff > 100) return 0

  // One conversion for up to four component scorers.
  const a = convSequence(p1)
  const b = convSequence(p2)

  // Measured after conversion: upstream counts Python characters, so a
  // character outside the BMP is one element and not the two UTF-16 code units
  // it occupies. Taking the ratio first would pick the partial-scoring branch
  // for lengths the component scorers below never see.
  const len1 = a.length
  const len2 = b.length
  const lenRatio = len1 > len2 ? len1 / len2 : len2 / len1

  let endRatio = ratioConverted(a, b, scoreCutoff)

  if (lenRatio < 1.5) {
    // With no token separator on either side, neither token-set nor token-sort
    // ratio can improve on the already-computed base ratio. Stated as "contains
    // no whitespace" rather than "splits into one token" because that is what
    // the test actually proves — a single token with a space around it splits
    // into one token too, but its sorted form differs from the input.
    if (!containsWhitespace(a) && !containsWhitespace(b)) return endRatio
    scoreCutoff = Math.max(scoreCutoff, endRatio) / UNBASE_SCALE
    return Math.max(endRatio, tokenRatioConverted(a, b, scoreCutoff) * UNBASE_SCALE)
  }

  const PARTIAL_SCALE = lenRatio <= 8 ? 0.9 : 0.6

  scoreCutoff = Math.max(scoreCutoff, endRatio) / PARTIAL_SCALE
  endRatio = Math.max(endRatio, partialRatioConverted(a, b, scoreCutoff) * PARTIAL_SCALE)

  scoreCutoff = Math.max(scoreCutoff, endRatio) / UNBASE_SCALE
  return Math.max(
    endRatio,
    partialTokenRatioConverted(a, b, scoreCutoff) * UNBASE_SCALE * PARTIAL_SCALE,
  )
}

/**
 * `ratio`, except that any comparison involving an empty processed input
 * answers `0`.
 *
 * The test is on either side being empty, which is what the code guarantees.
 * Only one of those cases actually parts company with {@link ratio_impl}: two
 * empty inputs are a perfect match to `ratio` and score `100`, where this scores
 * `0`. An empty against a non-empty already scores `0` both ways, since nothing
 * can be in common. "Processed" is load-bearing — a processor that strips its
 * input to nothing lands here too.
 */
export function qRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const p1 = applyProcessor(asSequence(s1), options.processor)
  const p2 = applyProcessor(asSequence(s2), options.processor)

  // FuzzyWuzzy returns 0 here; kept for compatibility. See RapidFuzz issue 110.
  if (p1.length === 0 || p2.length === 0) return 0

  return ratio_impl(p1, p2, { scoreCutoff: options.scoreCutoff })
}
