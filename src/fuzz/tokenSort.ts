import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import type { PatternMask } from '../algorithms/shared/bitmask/pattern.js'
import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  withChoicePreparer,
  FUZZ_FLAGS,
  type NormalizedScorer,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import { preparedTokenChoice, sortedOf, tokenChoicePreparer } from './internal/tokens.js'
import { tokenSortRatio_impl, tokenSortRatioConverted } from './internal/tokenSort.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

/** Narrow prepared-query implementation for token-sort similarity. */
export function prepareTokenSort(): PreparedScorerFactory {
  const choicePreparer = tokenChoicePreparer()
  const prepare: PrepareScorer = (query) => {
    const queryChoice = preparedTokenChoice(choicePreparer(query))
    const held = queryChoice.sequence
    let pattern: PatternMask | null = null
    const patternOf = (): PatternMask => {
      if (pattern === null) {
        const sorted = sortedOf(queryChoice)
        pattern = prepareLcsPattern(sorted, 0, sorted.length)
      }
      return pattern
    }
    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const choice = preparedTokenChoice(rawChoice)
      return tokenSortRatioConverted(
        held,
        choice.sequence,
        rawCutoff ?? 0,
        queryChoice,
        choice,
        patternOf(),
      )
    }
    return score
  }
  return withChoicePreparer(prepare, choicePreparer)
}

const BOUNDS: readonly [number, number] = [0, 100]

export const tokenSortRatio: NormalizedScorer<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(tokenSortRatio_impl, FUZZ_FLAGS, prepareTokenSort())

export const tokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
