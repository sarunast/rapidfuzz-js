import type { BuiltInMetric } from '../../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from '../metric.js'
import { prepareFuzz } from '../preparation.js'
import type { FuzzOptions } from '../types.js'
import { tokenSetRatio_impl } from './tokenSet.js'

export const tokenSetRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenSetRatio'),
  )

/**
 * Compares the two inputs as **sets of tokens** — deduplicated, sorted — and
 * factors out the tokens they share, `0..100`.
 *
 * Because the common tokens are removed before comparing, extra words on one
 * side cost far less than under any other scorer. It is the strongest tool in
 * the family for "same thing, described at different lengths":
 *
 * ```ts
 * tokenSetSimilarity('mariners vs angels', 'los angeles angels of anaheim at seattle mariners')
 * // 90.91… — where tokenSortSimilarity gives 50.74…
 * ```
 *
 * The consequence worth knowing, and the trap: whenever one token set
 * *contains* the other, the score is a flat `100`, however much extra the
 * longer side carries.
 *
 * ```ts
 * tokenSetSimilarity('data engineer', 'data engineer cloud platform') // 100
 * ```
 *
 * That is exactly right for a company name — "Hoval" against "Hoval AG" is one
 * employer — and exactly wrong for a job title, where the extra words are the
 * whole difference. Use `tokenSortSimilarity` when length should still count.
 *
 * Two inputs with no tokens between them score `0`, not `100`: an empty or
 * whitespace-only side has no set to intersect. FuzzyWuzzy answers `0` here and
 * RapidFuzz keeps it (issue 110), so this is compatibility rather than identity.
 *
 * RapidFuzz calls it `token_set_ratio`.
 */
export const tokenSetSimilarity: BuiltInMetric<'fuzz.tokenSetSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(tokenSetRatio)
