import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { partialTokenSetRatio_impl } from './internal/tokenSet.js'
import type { FuzzOptions } from './types.js'

export const partialTokenSetRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSetRatio'),
  )

export const partialTokenSetSimilarity: BuiltInMetric<
  'fuzz.partialTokenSetSimilarity',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(partialTokenSetRatio)
