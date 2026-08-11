import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { prepareFuzz } from './internal/prepared.js'
import { partialTokenSortRatio_impl } from './internal/tokenSet.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

const BOUNDS: readonly [number, number] = [0, 100]

export const partialTokenSortRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialTokenSortRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialTokenSortRatio'),
  )

export const partialTokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialTokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
