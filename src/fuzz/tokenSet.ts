import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { tokenSetRatio_impl } from './internal/tokenSet.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

const BOUNDS: readonly [number, number] = [0, 100]

export const tokenSetRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tokenSetRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('tokenSetRatio'),
  )

export const tokenSetSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenSetRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
