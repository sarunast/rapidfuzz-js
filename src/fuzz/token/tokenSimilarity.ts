import type { BuiltInMetric } from '../../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from '../internal/metric.js'
import { prepareFuzz } from '../internal/prepared.js'
import type { FuzzOptions } from '../types.js'
import { tokenRatio_impl } from './tokenSet.js'

export const tokenRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenRatio'),
  )

/**
 * The higher of `tokenSortSimilarity` and `tokenSetSimilarity`, `0..100`.
 *
 * For text that may differ in word order *or* in how much extra one side
 * carries, without your having to decide which up front. Because it takes the
 * maximum it inherits token-set's blind spot — containment scores `100` — so
 * where precision matters, name the strategy you actually want instead.
 *
 * RapidFuzz calls it `token_ratio`.
 */
export const tokenSimilarity: BuiltInMetric<'fuzz.tokenSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(tokenRatio)
