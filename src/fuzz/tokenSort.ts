import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import type { PatternMask } from '../algorithms/shared/bitmask/pattern.js'
import { builtInMetric } from '../algorithms/shared/metricAdapter.js'
import {
  type PreparationFactory,
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../algorithms/shared/scorerSupport.js'
import type { Metric } from '../core/metric.js'
import type { PreparedKernel } from '../core/protocol.js'
import type { Sequence } from '../core/types.js'
import { preparedTokenChoice, sortedOf, tokenChoicePreparer } from './internal/tokens.js'
import { tokenSortRatio_impl, tokenSortRatioConverted } from './internal/tokenSort.js'
import type { FuzzConfiguration, FuzzOptions } from './types.js'

/** Narrow prepared-query implementation for token-sort similarity. */
export function prepareTokenSort(): PreparationFactory {
  const choicePreparer = tokenChoicePreparer()
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const queryChoice = preparedTokenChoice(choicePreparer(query))
    let pattern: PatternMask | null = null
    const patternOf = (): PatternMask => {
      if (pattern === null) {
        const sorted = sortedOf(queryChoice)
        pattern = prepareLcsPattern(sorted, 0, sorted.length)
      }
      return pattern
    }
    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const choice = preparedTokenChoice(rawChoice)
      // `patternOf`, not `patternOf()`: the callee refuses an impossible cutoff
      // before it sorts anything, and an argument would have built the query's
      // masks on the way in regardless.
      return tokenSortRatioConverted(
        queryChoice.sequence,
        choice.sequence,
        rawCutoff ?? 0,
        queryChoice,
        choice,
        patternOf,
      )
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: choicePreparer })
}

const BOUNDS: readonly [number, number] = [0, 100]

export const tokenSortRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(tokenSortRatio_impl, FUZZ_FLAGS, prepareTokenSort())

export const tokenSortSimilarity: Metric<'similarity', FuzzConfiguration> =
  /* @__PURE__ */ builtInMetric({
    implementation: tokenSortRatio,
    direction: 'similarity',
    bounds: BOUNDS,
  })
