import type { PatternMask } from '../../algorithms/shared/bitmask/pattern.js'
import { validateSequence, isMissing } from '../../algorithms/shared/scorerSupport.js'
import { indelNormSimHeld, ratioConverted } from '../partialWindow.js'
import type { FuzzInput, FuzzOptions } from '../types.js'
import {
  canonicalLengthOf,
  sortedOf,
  tokenPair,
  tokenViewOf,
  type PreparedTokenChoice,
} from './tokens.js'

export function tokenSortRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  sortedPatternA?: () => PatternMask,
): number {
  if (scoreCutoff > 100) return 0

  const choiceA = viewA ?? tokenViewOf(a)
  const choiceB = viewB ?? tokenViewOf(b)

  if (scoreCutoff > 0) {
    const lengthA = canonicalLengthOf(choiceA)
    const lengthB = canonicalLengthOf(choiceB)
    const maximum = lengthA + lengthB
    if (maximum === 0) return 100
    if (1 - Math.abs(lengthA - lengthB) / maximum < scoreCutoff / 100) return 0
  }

  const sortedA = sortedOf(choiceA)
  const sortedB = sortedOf(choiceB)

  if (sortedPatternA !== undefined) {
    return (
      indelNormSimHeld(
        sortedPatternA(),
        sortedA.length,
        sortedB,
        0,
        sortedB.length,
        scoreCutoff / 100,
      ) * 100
    )
  }

  return ratioConverted(sortedA, sortedB, scoreCutoff)
}

export function tokenSortRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  return tokenSortRatioConverted(a, b, options.scoreCutoff ?? 0)
}
