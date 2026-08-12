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

/**
 * Two inputs with no tokens between them score `0`, not `100`: an empty or
 * whitespace-only side has no set to intersect. FuzzyWuzzy answers `0` here and
 * RapidFuzz keeps it (issue 110), so this is compatibility rather than identity.
 */
export const tokenSetSimilarity: BuiltInMetric<'fuzz.tokenSetSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(tokenSetRatio)
