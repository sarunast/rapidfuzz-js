import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { tokenRatio_impl } from './internal/tokenSet.js'
import type { FuzzOptions } from './types.js'

export const tokenRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenRatio'),
  )

export const tokenSimilarity: BuiltInMetric<'fuzz.tokenSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(tokenRatio)
