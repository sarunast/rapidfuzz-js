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

/**
 * `tokenSortSimilarity` over the best partial window, `0..100`: tokens are
 * sorted, then the shorter result is slid across the longer.
 *
 * For text that is both reordered and embedded in something larger — a scrambled
 * name inside a longer record.
 *
 * ```ts
 * partialTokenSortSimilarity('mariners vs angels', 'los angeles angels of anaheim at seattle mariners')
 * // 72.22… — where plain tokenSort gives 50.74…
 * ```
 *
 * RapidFuzz calls it `partial_token_sort_ratio`.
 */
export const partialTokenSortSimilarity: BuiltInMetric<
  'fuzz.partialTokenSortSimilarity',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(partialTokenSortRatio)
