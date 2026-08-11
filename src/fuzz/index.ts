import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import type { Metric } from '../core/metric.js'
import type { SimilarityConfiguration } from '../core/types.js'
import {
  partialRatio,
  partialRatioAlignment,
  partialTokenRatio,
  partialTokenSetRatio,
  partialTokenSortRatio,
  ratio,
  tokenRatio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
} from './internal/scorers.js'
import type { FuzzInput, ScoreAlignment } from './types.js'

export type { ScoreAlignment } from './types.js'
export type FuzzConfiguration = SimilarityConfiguration

const BOUNDS: readonly [number, number] = [0, 100]

export const similarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: ratio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSetSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenSetRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialTokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSetSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialTokenSetRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: partialTokenRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const fuzzySimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: wRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })

export function partialSimilarityAlignment(
  a: FuzzInput,
  b: FuzzInput,
): ScoreAlignment | null {
  return partialRatioAlignment(a, b)
}
