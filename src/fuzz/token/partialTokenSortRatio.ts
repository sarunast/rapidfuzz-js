import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../core/scoring/builtIn/implementation.js'
import type { BuiltInMetric } from '../../core/scoring/builtIn/metric.js'
import { fuzzMetric } from '../metric.js'
import { prepareFuzz } from '../preparation.js'
import type { FuzzOptions } from '../types.js'
import { partialTokenSortRatio_impl } from './tokenSet.js'

export const fuzzPartialTokenSortRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSortRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSortRatio'),
  )

/**
 * `tokenSortRatio` over the best partial window, `0..100`: tokens are
 * sorted, then the shorter result is slid across the longer.
 *
 * For text that is both reordered and embedded in something larger — a scrambled
 * name inside a longer record.
 *
 * ```ts
 * partialTokenSortRatio('mariners vs angels', 'los angeles angels of anaheim at seattle mariners')
 * // 72.22… — where plain tokenSort gives 50.74…
 * ```
 *
 * RapidFuzz spells it `partial_token_sort_ratio`.
 */
export const partialTokenSortRatio: BuiltInMetric<
  'fuzz.partialTokenSortRatio',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(fuzzPartialTokenSortRatio)
