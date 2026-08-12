import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { partialTokenRatio_impl } from './internal/tokenSet.js'
import type { FuzzOptions } from './types.js'

export const partialTokenRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenRatio'),
  )

export const partialTokenSimilarity: BuiltInMetric<
  'fuzz.partialTokenSimilarity',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(partialTokenRatio)
