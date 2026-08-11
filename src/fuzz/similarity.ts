import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  prepareScorerChoice,
  preparedScorerSequence,
  withChoicePreparer,
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { ratioHeld, ratio_impl } from './internal/partialWindow.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

/** Narrow prepared-query implementation for the basic fuzz similarity. */
export function prepareSimilarity(): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const held = preparedScorerSequence(prepareScorerChoice(query))
    const pattern = prepareLcsPattern(held, 0, held.length)
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const choice = preparedScorerSequence(rawChoice)
      return ratioHeld(pattern, held.length, choice, rawCutoff ?? 0)
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

const BOUNDS: readonly [number, number] = [0, 100]

export const ratio: NormalizedScorer<FuzzOptions> = /* @__PURE__ */ withPreparedFlags(
  ratio_impl,
  FUZZ_FLAGS,
  prepareSimilarity(),
)

export const similarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: ratio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
