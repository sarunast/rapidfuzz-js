import { prepareLcsPattern } from '../../algorithms/lcs/implementation.js'
import type { PatternMask } from '../../algorithms/shared/bitmask/index.js'
import {
  isNone,
  isSequence,
  type PrepareScorer,
  type PreparedScore,
  withChoicePreparer,
} from '../../algorithms/shared/scorerSupport.js'
import { preparedTokenChoice, sortedOf, tokenChoicePreparer } from './tokens.js'
import { tokenSortRatioConverted } from './tokenScorers.js'

/** Narrow prepared-query implementation for token-sort similarity. */
export function prepareTokenSort(): PrepareScorer {
  const choicePreparer = tokenChoicePreparer()
  const prepare: PrepareScorer = (query) => {
    const queryChoice = preparedTokenChoice(choicePreparer(query))
    if (queryChoice === null) throw new TypeError('fuzz scorers expect a sequence')
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
      if (isNone(rawChoice)) return 0
      let choice = preparedTokenChoice(rawChoice)
      if (choice === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('fuzz scorers expect a string or an array-like sequence')
        }
        choice = preparedTokenChoice(choicePreparer(rawChoice))
      }
      if (choice === null) throw new TypeError('fuzz scorers expect a sequence')
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
