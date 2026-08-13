import type { BuiltInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import { fuzzMetric } from './metric.js'
import { partialRatioAlignment, partialRatio_impl } from './partialWindow.js'
import { prepareFuzz } from './preparation.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'

export const partialRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(
    partialRatio_impl,
    FUZZ_FLAGS,
    prepareFuzz('partialRatio'),
  )

/**
 * Slides the shorter input across the longer and reports the best-scoring window,
 * `0..100` — so a substring scores `100` however different the surrounding text.
 *
 * ```ts
 * partialSimilarity('new york jets', 'the new york jets play tonight') // 100
 * similarity('new york jets', 'the new york jets play tonight') // 60.46…
 * ```
 *
 * Use it when one side may be contained in the other: a search box query against a
 * full title, a name against a sentence. It says nothing about word *order* within
 * the window — `partialSimilarity('smith john', 'john smith')` is only `66.67` —
 * so pair it with a token strategy when both problems are present.
 *
 * {@link partialSimilarityAlignment} returns the same score plus where the window
 * sat.
 *
 * RapidFuzz calls it `partial_ratio`.
 */
export const partialSimilarity: BuiltInMetric<'fuzz.partialSimilarity', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(partialRatio)

/**
 * {@link partialSimilarity} with the window it chose — the score plus the
 * `[srcStart, srcEnd)` and `[destStart, destEnd)` ranges it aligned, for
 * highlighting what actually matched.
 *
 * The one fuzz entry point that explains itself rather than only scoring.
 *
 * @returns `null` when either input is missing.
 */
export function partialSimilarityAlignment(
  a: FuzzInput,
  b: FuzzInput,
): ScoreAlignment | null {
  return partialRatioAlignment(a, b)
}

export { partialRatioAlignment }
