import type { BuiltInMetric } from '../../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from '../metric.js'
import { prepareFuzz } from '../preparation.js'
import type { FuzzOptions } from '../types.js'
import { partialTokenSetRatio_impl } from './tokenSet.js'

export const fuzzPartialTokenSetRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSetRatio'),
  )

/**
 * `tokenSetRatio` over the best partial window, `0..100`.
 *
 * The most forgiving scorer in the family: it already ignores word order and
 * extra words through the token set, and then slides for containment on top.
 * In practice it returns `100` for a very wide range of pairs, so treat a high
 * score here as weak evidence — it is most useful as a recall-oriented first
 * pass, with a stricter scorer deciding.
 *
 * As with `tokenSetRatio`, two tokenless inputs score `0` rather than
 * `100` — FuzzyWuzzy's answer, kept by RapidFuzz (issue 110).
 *
 * RapidFuzz spells it `partial_token_set_ratio`.
 */
export const partialTokenSetRatio: BuiltInMetric<
  'fuzz.partialTokenSetRatio',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(fuzzPartialTokenSetRatio)
