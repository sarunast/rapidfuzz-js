import type { BuiltInMetric } from '../../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from '../internal/metric.js'
import { prepareFuzz } from '../internal/prepared.js'
import type { FuzzOptions } from '../types.js'
import { partialTokenRatio_impl } from './tokenSet.js'

export const partialTokenRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenRatio'),
  )

/**
 * The higher of `partialTokenSortSimilarity` and `partialTokenSetSimilarity`,
 * `0..100` — the partial-window counterpart to `tokenSimilarity`.
 *
 * The most permissive scorer in the family after `weightedSimilarity`: it
 * forgives word order, extra words, and containment all at once, which makes it
 * prone to scoring unrelated pairs highly. Prefer a narrower scorer when you can
 * name the problem.
 *
 * RapidFuzz calls it `partial_token_ratio`.
 */
export const partialTokenSimilarity: BuiltInMetric<
  'fuzz.partialTokenSimilarity',
  'similarity'
> = /* @__PURE__ */ fuzzMetric(partialTokenRatio)
