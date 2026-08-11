import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { partialRatioAlignment, partialRatio_impl } from './internal/partialWindow.js'
import { prepareFuzz } from './internal/prepared.js'
import type {
  FuzzConfiguration,
  FuzzInput,
  FuzzOptions,
  ScoreAlignment,
} from './types.js'

const BOUNDS: readonly [number, number] = [0, 100]

export const partialRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialRatio'),
  )

export const partialSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })

export function partialSimilarityAlignment(
  a: FuzzInput,
  b: FuzzInput,
): ScoreAlignment | null {
  return partialRatioAlignment(a, b)
}

export { partialRatioAlignment }
