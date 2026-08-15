import {
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { BuiltInMetric } from '../core/scoring/builtIn/metric.js'
import { fuzzMetric } from './metric.js'
import { partialRatioAlignment_impl, partialRatio_impl } from './partialWindow.js'
import { prepareFuzz } from './preparation.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'

export const fuzzPartialRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
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
 * partialRatio('new york jets', 'the new york jets play tonight') // 100
 * similarity('new york jets', 'the new york jets play tonight') // 60.46…
 * ```
 *
 * Use it when one side may be contained in the other: a search box query against a
 * full title, a name against a sentence. It says nothing about word *order* within
 * the window — `partialRatio('smith john', 'john smith')` is only `66.67` —
 * so pair it with a token strategy when both problems are present.
 *
 * {@link partialRatioAlignment} returns the same score plus where the window
 * sat.
 *
 * RapidFuzz spells it `partial_ratio`.
 */
export const partialRatio: BuiltInMetric<'fuzz.partialRatio', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(fuzzPartialRatio)

/**
 * {@link partialRatio} with the window it chose — the score plus the
 * `[srcStart, srcEnd)` and `[destStart, destEnd)` ranges it aligned, for
 * highlighting what actually matched.
 *
 * The one fuzz entry point that explains itself rather than only scoring.
 *
 * @returns `null` when either input is missing.
 */
export function partialRatioAlignment(a: FuzzInput, b: FuzzInput): ScoreAlignment | null {
  return partialRatioAlignment_impl(a, b)
}
