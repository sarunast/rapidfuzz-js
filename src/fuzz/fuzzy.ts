import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
/**
 * `wRatio` picks a strategy rather than defining a new comparison algorithm.
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
import { fuzzMetric } from './internal/metric.js'
import { partialRatioConverted, ratioConverted } from './internal/partialWindow.js'
import { prepareFuzz } from './internal/prepared.js'
import {
  containsWhitespace,
  stringContainsWhitespace,
  tokenForm,
} from './internal/tokens.js'
import { partialTokenRatioConverted, tokenRatioConverted } from './internal/tokenSet.js'
import type { FuzzInput, FuzzOptions } from './types.js'

/**
 * Weighted combination of the other scorers, picking a strategy from the length
 * ratio of the two inputs. Misordered full matches are scaled by 0.95, and
 * partial matches by 0.9 up to an eightfold length difference and 0.6 past it.
 */
export function wRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const UNBASE_SCALE = 0.95

  const p1 = asSequence(s1)
  const p2 = asSequence(s2)

  // FuzzyWuzzy returns 0 here; kept for compatibility. See RapidFuzz issue 110.
  if (p1.length === 0 || p2.length === 0) return 0

  let scoreCutoff = options.scoreCutoff ?? 0
  if (scoreCutoff > 100) return 0

  // One conversion for up to four component scorers — but only when one is
  // needed. Almost every call is two BMP strings that share no whitespace, and
  // that call returns the base ratio below without a token scorer ever running;
  // converting first spent two `Uint32Array` allocations and two full scans to
  // reach an answer the strings could have given.
  //
  // The representation is a property of the *pair*, not of either side: keeping
  // a BMP string as a string while its partner became code points would have
  // them meet as `'a'` and `97`, which `===` reports as different. So this is
  // the same four-part test as `convPair`, in the same order, written out to
  // skip the tuple it allocates — the shape `levenshteinDistance_impl` already
  // uses, and what `hasSurrogatePair` is exported for.
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

  // Measured on whichever form the pair settled into, never on the raw input:
  // upstream counts Python characters, so a character outside the BMP is one
  // element and not the two UTF-16 code units it occupies. Taking the ratio
  // first would pick the partial-scoring branch for lengths the component
  // scorers below never see. The string branch above is not an exception —
  // it is reached only when neither side splits a code point, which is exactly
  // when `.length` already *is* the character count.
  const len1 = a.length
  const len2 = b.length
  const lenRatio = len1 > len2 ? len1 / len2 : len2 / len1

  let endRatio = ratioConverted(a, b, scoreCutoff)

  if (lenRatio < 1.5) {
    // Raised before the whitespace scans rather than after them. Every scorer
    // below answers 0 to a cutoff above 100, so once the base ratio puts the
    // token component out of reach — `endRatio` above 95, or a caller asking for
    // it — `Math.max` returns `endRatio` whatever runs. Testing that here rather
    // than at the `Math.max` skips two scans and both token conversions, which
    // is the common shape in best-match search: the running best is fed back as
    // the next cutoff, and it climbs.
    scoreCutoff = Math.max(scoreCutoff, endRatio) / UNBASE_SCALE
    if (scoreCutoff > 100) return endRatio

    // With no token separator on either side, neither token-set nor token-sort
    // ratio can improve on the already-computed base ratio. Stated as "contains
    // no whitespace" rather than "splits into one token" because that is what
    // the test actually proves — a single token with a space around it splits
    // into one token too, but its sorted form differs from the input.
    // Each side is tested through whichever of the two the pair settled into.
    // Both are strings or neither is, but only a test on each proves that to
    // the checker, and the pair of `typeof`s costs nothing beside the scan.
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
  // Same exit, one scale earlier: a cutoff past 100 here rules out the partial
  // scorer *and* the token one after it, whose cutoff is this number divided
  // again. Skipping straight to `endRatio` saves the window scan as well as the
  // two token conversions.
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

export const wRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(wRatio_impl, FUZZ_FLAGS, prepareFuzz('wRatio'))

export const fuzzySimilarity: BuiltInMetric<'fuzz.fuzzySimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(wRatio)
