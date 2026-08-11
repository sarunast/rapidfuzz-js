import { builtInMetric, type Metric, type SimilarityConfiguration } from './_metric.js'
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
} from './_fuzz/legacy.js'
import type { FuzzInput, ScoreAlignment } from './_fuzz/types.js'

export type { ScoreAlignment } from './_fuzz/types.js'
export type FuzzConfiguration = SimilarityConfiguration

const BOUNDS: readonly [number, number] = [0, 100]

export const similarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({ legacy: ratio, direction: 'similarity', bounds: BOUNDS })
export const partialSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: partialRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: tokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSetSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: tokenSetRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const tokenSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: tokenRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: partialTokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSetSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: partialTokenSetRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const partialTokenSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    legacy: partialTokenRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
export const fuzzySimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({ legacy: wRatio, direction: 'similarity', bounds: BOUNDS })

export function partialSimilarityAlignment(
  a: FuzzInput,
  b: FuzzInput,
): ScoreAlignment | null {
  return partialRatioAlignment(a, b)
}
