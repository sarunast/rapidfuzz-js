import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../core/scoring/builtIn/implementation.js'
import type { BuiltInMetric } from '../../core/scoring/builtIn/metric.js'
import { fuzzMetric } from '../metric.js'
import { prepareFuzz } from '../preparation.js'
import type { FuzzOptions } from '../types.js'
import { partialTokenRatio_impl } from './tokenSet.js'

export const fuzzPartialTokenRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenRatio'),
  )

/**
 * The higher of `partialTokenSortRatio` and `partialTokenSetRatio`,
 * `0..100` — the partial-window counterpart to `tokenRatio`.
 *
 * The most permissive scorer in the family after `weightedRatio`: it
 * forgives word order, extra words, and containment all at once, which makes it
 * prone to scoring unrelated pairs highly. Prefer a narrower scorer when you can
 * name the problem.
 *
 * RapidFuzz spells it `partial_token_ratio`.
 */
export const partialTokenRatio: BuiltInMetric<'fuzz.partialTokenRatio', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(fuzzPartialTokenRatio)
