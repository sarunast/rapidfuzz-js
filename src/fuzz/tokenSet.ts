import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { tokenSetRatio_impl } from './internal/tokenSet.js'
import type { FuzzOptions } from './types.js'

export const tokenSetRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenSetRatio'),
  )

export const tokenSetSimilarity: BuiltInMetric<'fuzz.tokenSetSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(tokenSetRatio)
