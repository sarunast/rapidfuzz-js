import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { partialTokenRatio_impl } from './internal/tokenSet.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

const BOUNDS: readonly [number, number] = [0, 100]

export const partialTokenRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenRatio'),
  )

export const partialTokenSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialTokenRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
