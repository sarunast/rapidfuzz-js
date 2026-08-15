import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
/**
 * `weightedRatio` picks a strategy rather than defining a new comparison algorithm.
 *
 * It chooses among `ratio`, `tokenRatio`,
 * `partialRatio` and `partialTokenRatio` from the length ratio of its two
 * inputs and scales what it gets by 0.95, 0.9 or 0.6.
 *
 * This is the top *raw-input* layer of the scorer graph: it imports `basic`,
 * `tokens` and `tokenScorers`, and only the facade imports it. `prepared.ts`
 * does **not** — it mirrors this strategy independently over prepared
 * state, because everything an `*_impl` here does first, validating and
 * converting raw input, is already done by the time a prepared branch runs. The
 * two copies have to be kept in step by hand.
 */
import {
  asSequence,
  convSequence,
  hasSurrogatePair,
  isMissing,
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './metric.js'
import { partialRatioConverted, ratioConverted } from './partialWindow.js'
import { prepareFuzz } from './preparation.js'
import {
  containsWhitespace,
  stringContainsWhitespace,
  tokenForm,
} from './token/tokens.js'
import { partialTokenRatioConverted, tokenRatioConverted } from './token/tokenSet.js'
import type { FuzzInput, FuzzOptions } from './types.js'

/**
 * Weighted combination of the other scorers, picking a strategy from the length
 * ratio of the two inputs. Misordered full matches are scaled by 0.95, and
 * partial matches by 0.9 up to an eightfold length difference and 0.6 past it.
 */
export function weightedRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const UNBASE_SCALE = 0.95

  const p1 = asSequence(s1)
  const p2 = asSequence(s2)

  if (p1.length === 0 || p2.length === 0) return 0

  let scoreCutoff = options.scoreCutoff ?? 0
  if (scoreCutoff > 100) return 0

  let a: ArrayLike<unknown>
  let b: ArrayLike<unknown>
  if (
    typeof p1 === 'string' &&
    typeof p2 === 'string' &&
    !hasSurrogatePair(p1) &&
    !hasSurrogatePair(p2)
  ) {
    a = p1
    b = p2
  } else {
    a = convSequence(p1)
    b = convSequence(p2)
  }

  const len1 = a.length
  const len2 = b.length
  const lenRatio = len1 > len2 ? len1 / len2 : len2 / len1

  let endRatio = ratioConverted(a, b, scoreCutoff)

  if (lenRatio < 1.5) {
    scoreCutoff = Math.max(scoreCutoff, endRatio) / UNBASE_SCALE
    if (scoreCutoff > 100) return endRatio

    if (
      !(typeof a === 'string' ? stringContainsWhitespace(a) : containsWhitespace(a)) &&
      !(typeof b === 'string' ? stringContainsWhitespace(b) : containsWhitespace(b))
    ) {
      return endRatio
    }
    return Math.max(
      endRatio,
      tokenRatioConverted(tokenForm(a), tokenForm(b), scoreCutoff) * UNBASE_SCALE,
    )
  }

  const PARTIAL_SCALE = lenRatio <= 8 ? 0.9 : 0.6

  scoreCutoff = Math.max(scoreCutoff, endRatio) / PARTIAL_SCALE
  if (scoreCutoff > 100) return endRatio

  endRatio = Math.max(endRatio, partialRatioConverted(a, b, scoreCutoff) * PARTIAL_SCALE)

  scoreCutoff = Math.max(scoreCutoff, endRatio) / UNBASE_SCALE
  if (scoreCutoff > 100) return endRatio

  return Math.max(
    endRatio,
    partialTokenRatioConverted(tokenForm(a), tokenForm(b), scoreCutoff) *
      UNBASE_SCALE *
      PARTIAL_SCALE,
  )
}

export const fuzzWeightedRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    weightedRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('weightedRatio'),
  )

/**
 * Picks the strategies that suit the pair, scales each by how much it can be
 * trusted for those lengths, and reports the best result, `0..100`.
 *
 * The adaptive one, and the right default. It always computes `similarity` as a
 * base, then picks what else to try from the ratio of the two lengths:
 *
 * - **below 1.5** — the inputs are comparable, so it also tries
 *   `tokenRatio`, scaled by `0.95`. With no whitespace on either side there
 *   are no tokens to reorder, and it stops at the base.
 * - **1.5 and above** — one side is much longer, so it tries `partialRatio`
 *   and `partialTokenRatio` instead, scaled by `0.9` — or by `0.6` once the
 *   longer side is more than eight times the shorter, where a best window means
 *   much less.
 *
 * The result is the **maximum**, not a consensus: the scale factors are what
 * keep a broader strategy honest, since a partial match has to beat the
 * whole-string score by more than a tenth before it can win.
 *
 * ```ts
 * weightedRatio('smith john', 'john smith') // 95
 * weightedRatio('new york jets', 'the new york jets play tonight') // 90
 * weightedRatio('this is a test', 'this is a test!') // 96.55…
 * ```
 *
 * Start here, look at the pairs it gets wrong on your data, and only pin a
 * specific scorer once you can name the failure. What you give up is
 * explainability: the number does not tell you which strategy produced it.
 *
 * Two empty inputs score `0`, not `100` — FuzzyWuzzy's answer, kept by
 * RapidFuzz (issue 110). Whitespace-only inputs still have length, so those
 * reach the ordinary strategy and score `100`.
 *
 * RapidFuzz spells it `WRatio`.
 */
export const weightedRatio: BuiltInMetric<'fuzz.weightedRatio', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(fuzzWeightedRatio)
