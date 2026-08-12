import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { partialTokenSortRatio_impl } from './internal/tokenSet.js'
import type { FuzzOptions } from './types.js'

export const partialTokenSortRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSortRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSortRatio'),
  )

export const partialTokenSortSimilarity: BuiltInMetric<
  'fuzz.partialTokenSortSimilarity',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(partialTokenSortRatio)
