import { prepareLcsPattern } from '../../algorithms/lcs/implementation.js'
import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
import type { BuiltInMetric } from '../../algorithms/shared/metricAdapter.js'
import {
  type PreparationFactory,
  FUZZ_FLAGS,
  type MaybeSequenceMetricImplementation,
  withPreparedFlags,
} from '../../algorithms/shared/scorerSupport.js'
import type { PreparedKernel } from '../../core/scoring/compilation.js'
import type { Sequence } from '../../core/types.js'
import { fuzzMetric } from '../metric.js'
import type { FuzzOptions } from '../types.js'
import { preparedTokenChoice, sortedOf, tokenChoicePreparer } from './tokens.js'
import { tokenSortRatio_impl, tokenSortRatioConverted } from './tokenSort.js'

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

export const fuzzTokenSortRatio: MaybeSequenceMetricImplementation<FuzzOptions> =
  /* @__PURE__ */ withPreparedFlags(tokenSortRatio_impl, FUZZ_FLAGS, prepareTokenSort())

/**
 * Splits both inputs into tokens, sorts them, and compares the sorted results,
 * `0..100` — so word order stops mattering while extra words still count.
 *
 * ```ts
 * tokenSortRatio('smith john', 'john smith') // 100
 * tokenSortRatio('data engineer', 'data engineer cloud platform') // 63.41…
 * ```
 *
 * That second line is the reason to choose this over `tokenSetRatio`: it
 * stays length-aware, so a longer string with extra words is *not* a perfect
 * match. For a job title or a
 * product name, where the extra words are the difference, that is the behaviour
 * you want.
 *
 * RapidFuzz spells it `token_sort_ratio`.
 */
export const tokenSortRatio: BuiltInMetric<'fuzz.tokenSortRatio', 'similarity'> =
  /* @__PURE__ */ fuzzMetric(fuzzTokenSortRatio)
