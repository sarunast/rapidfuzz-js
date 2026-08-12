import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './internal/metric.js'
import { partialRatioAlignment, partialRatio_impl } from './internal/partialWindow.js'
import { prepareFuzz } from './internal/prepared.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'

export const partialRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialRatio'),
  )

export const partialSimilarity: BuiltInMetric<'fuzz.partialSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(partialRatio)

export function partialSimilarityAlignment(
  a: FuzzInput,
  b: FuzzInput,
): ScoreAlignment | null {
  return partialRatioAlignment(a, b)
}

export { partialRatioAlignment }
