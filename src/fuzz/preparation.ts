import { prepareLcsPattern } from '../algorithms/lcs/implementation.js'
import type { PatternMask } from '../algorithms/shared/bitmask/pattern.js'
import {
  alignRepresentation,
  convSequence,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type ChoicePreparer,
  type PreparationFactory,
} from '../algorithms/shared/scorerSupport.js'
import type { PreparedKernel } from '../core/scoring/compilation.js'
import type { Sequence } from '../core/types.js'
import {
  type CharSet,
  charSetOf,
  partialAlignmentConverted,
  partialRatioConverted,
  partialRatioImpl,
  ratioHeld,
} from './partialWindow.js'
import { tokenContainmentProof } from './token/containment.js'
import {
  hasWhitespaceOf,
  preparedTokenChoice,
  sortedOf,
  tokenChoicePreparer,
  tokenViewOf,
} from './token/tokens.js'
import {
  partialTokenRatioConverted,
  partialTokenSetRatioConverted,
  tokenRatioConverted,
  tokenSetRatioConverted,
} from './token/tokenSet.js'
import type { PreparedFuzzKind } from './types.js'

function tokenisesInput(kind: PreparedFuzzKind): boolean {
  switch (kind) {
    case 'partialRatio':
      return false
    case 'partialTokenSortRatio':
    case 'tokenSetRatio':
    case 'partialTokenSetRatio':
    case 'tokenRatio':
    case 'partialTokenRatio':
    case 'weightedRatio':
      return true
  }
}

export function prepareFuzz(kind: PreparedFuzzKind): PreparationFactory {
  const usesTokens = tokenisesInput(kind)
  const choicePreparer: ChoicePreparer = usesTokens
    ? tokenChoicePreparer()
    : prepareChoiceSequence

  const prepareQuery = (query: Sequence): PreparedKernel => {
    const queryTokenChoice = usesTokens
      ? preparedTokenChoice(choicePreparer(query))
      : null
    const heldQuery =
      queryTokenChoice === null ? scorerSequence(query) : queryTokenChoice.sequence
    const a = heldQuery
    let lcsPattern: PatternMask | null = null
    const patternOf = (): PatternMask =>
      (lcsPattern ??= prepareLcsPattern(a, 0, a.length))
    const queryView = queryTokenChoice ?? undefined
    const queryTokens = queryTokenChoice ?? tokenViewOf(a)
    let sortedPattern: PatternMask | null = null
    const sortedPatternOf = (): PatternMask => {
      if (sortedPattern === null) {
        const sorted = sortedOf(queryTokens)
        sortedPattern = prepareLcsPattern(sorted, 0, sorted.length)
      }
      return sortedPattern
    }
    let sortedCharSet: CharSet | null = null
    const sortedCharSetOf = (): CharSet =>
      (sortedCharSet ??= charSetOf(sortedOf(queryTokens)))
    let nativeCharSet: CharSet | null = null
    const nativeCharSetOf = (): CharSet => (nativeCharSet ??= charSetOf(a))
    let convertedCharSet: CharSet | null = null
    const convertedCharSetOf = (): CharSet =>
      (convertedCharSet ??= charSetOf(convSequence(a)))

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const cutoff = rawCutoff ?? 0

      switch (kind) {
        case 'partialRatio': {
          const b = preparedChoiceSequence(rawChoice)
          const s1 = alignRepresentation(a, b)
          const s2 = alignRepresentation(b, a)
          return (
            partialAlignmentConverted(
              s1,
              s2,
              cutoff,
              true,
              patternOf(),
              s1 === a ? nativeCharSetOf() : convertedCharSetOf(),
            )?.score ?? 0
          )
        }
        case 'tokenSetRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return tokenSetRatioConverted(a, choice.sequence, cutoff, queryView, choice)
        }
        case 'tokenRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return tokenRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
            sortedPatternOf,
          )
        }
        case 'partialTokenSortRatio': {
          const choice = preparedTokenChoice(rawChoice)
          const sortedQuery = sortedOf(queryTokens)
          const sortedChoice = sortedOf(choice)
          const preparedApplies = sortedQuery.length <= sortedChoice.length

          return (
            partialAlignmentConverted(
              sortedQuery,
              sortedChoice,
              cutoff,
              true,
              preparedApplies ? sortedPatternOf() : undefined,
              preparedApplies ? sortedCharSetOf() : undefined,
            )?.score ?? 0
          )
        }
        case 'partialTokenSetRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return partialTokenSetRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
          )
        }
        case 'partialTokenRatio': {
          const choice = preparedTokenChoice(rawChoice)
          return partialTokenRatioConverted(
            a,
            choice.sequence,
            cutoff,
            queryView,
            choice,
            sortedPatternOf,
            sortedCharSetOf,
          )
        }
        case 'weightedRatio': {
          const preparedTokens = preparedTokenChoice(rawChoice)
          const b = preparedTokens.sequence
          if (a.length === 0 || b.length === 0 || cutoff > 100) return 0
          const unbaseScale = 0.95
          const lenRatio = a.length > b.length ? a.length / b.length : b.length / a.length
          let dynamicCutoff = cutoff
          let result = ratioHeld(patternOf(), a.length, b, dynamicCutoff)

          if (lenRatio < 1.5) {
            dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
            if (dynamicCutoff > 100) return result

            if (!hasWhitespaceOf(queryTokens) && !hasWhitespaceOf(preparedTokens))
              return result
            return Math.max(
              result,
              tokenRatioConverted(
                a,
                b,
                dynamicCutoff,
                queryView,
                preparedTokens,
                sortedPatternOf,
              ) * unbaseScale,
            )
          }

          const partialScale = lenRatio <= 8 ? 0.9 : 0.6
          dynamicCutoff = Math.max(dynamicCutoff, result) / partialScale
          if (dynamicCutoff > 100) return result

          const partial =
            a.length <= b.length
              ? partialRatioImpl(
                  a,
                  b,
                  dynamicCutoff / 100,
                  patternOf(),
                  true,
                  nativeCharSetOf(),
                ).score
              : partialRatioConverted(a, b, dynamicCutoff)
          result = Math.max(result, partial * partialScale)
          dynamicCutoff = Math.max(dynamicCutoff, result) / unbaseScale
          if (dynamicCutoff > 100) return result

          return Math.max(
            result,
            partialTokenRatioConverted(
              a,
              b,
              dynamicCutoff,
              queryView,
              preparedTokens,
              sortedPatternOf,
              sortedCharSetOf,
            ) *
              unbaseScale *
              partialScale,
          )
        }
      }
    }
    return score
  }
  const proveOptimum = kind === 'tokenSetRatio' ? tokenContainmentProof : undefined

  return () => ({ prepareQuery, prepareChoice: choicePreparer, proveOptimum })
}
