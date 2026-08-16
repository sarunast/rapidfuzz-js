import type { PatternMask } from '#algorithms/bitmask/pattern.js'
import {
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  lcsSeqLengthRange,
} from '#algorithms/lcs/implementation.js'
import { validateSequence, isMissing } from '#core/sequence.js'

import {
  type CharSet,
  partialAlignmentConverted,
  partialRatioConverted,
} from '../partialWindow.js'
import type { FuzzInput, FuzzOptions } from '../types.js'
import {
  difference,
  intersects,
  joinTokens,
  sortedOf,
  sortedUniqueOf,
  tokenPair,
  splitOf,
  tokenViewOf,
  uniqueJoinedOf,
  uniqueOf,
  type PreparedTokenChoice,
  type UniqueTokenSet,
} from './tokens.js'
import { tokenSortRatioConverted } from './tokenSort.js'

function normDistance(dist: number, lensum: number, scoreCutoff: number): number {
  const score = 100 - (100 * dist) / lensum
  return score >= scoreCutoff ? score : 0
}

function indelDist(a: ArrayLike<unknown>, b: ArrayLike<unknown>, budget: number): number {
  return (
    a.length + b.length - 2 * lcsSeqLengthRange(a, 0, a.length, b, 0, b.length, budget)
  )
}

export function tokenSetRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  return tokenSetRatioConverted(a, b, options.scoreCutoff ?? 0)
}

function disjointTokensRatio(
  abLen: number,
  patternA: () => PatternMask,
  viewB: PreparedTokenChoice,
  tokensB: UniqueTokenSet,
  scoreCutoff: number,
): number {
  const baLen = tokensB.payload + tokensB.size - 1
  const lensum = abLen + baLen
  const cutoffDistance = Math.ceil(lensum * (1 - scoreCutoff / 100))
  const lengthDiff = abLen > baLen ? abLen - baLen : baLen - abLen
  if (lengthDiff > cutoffDistance) return 0

  const joinedB = uniqueJoinedOf(viewB)
  const required = Math.ceil((lensum - cutoffDistance) / 2)
  const lcs =
    required > 0 && scoreCutoff >= 70 && lensum >= 128
      ? lcsSeqLengthPreparedBounded(patternA(), joinedB, 0, baLen, required)
      : lcsSeqLengthPrepared(patternA(), joinedB, 0, baLen)
  if (lcs < 0) return 0

  const dist = lensum - 2 * lcs
  if (dist > cutoffDistance) return 0

  return normDistance(dist, lensum, scoreCutoff)
}

export function tokenSetRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  uniquePatternA?: () => PatternMask,
): number {
  if (scoreCutoff > 100) return 0

  const tokensA = uniqueOf(viewA ?? tokenViewOf(a))
  const tokensB = uniqueOf(viewB ?? tokenViewOf(b))

  if (tokensA.size === 0 || tokensB.size === 0) return 0

  let sectCount = 0
  let sectPayload = 0
  let diffAbPayload = 0
  let diffBaPayload = 0
  const diffAb: unknown[][] = []
  const diffBa: unknown[][] = []

  const entriesA = sortedUniqueOf(tokensA)
  for (let i = 0; i < entriesA.keys.length; i++) {
    const token = entriesA.tokens[i]
    if (tokensB.has(entriesA.keys[i], token)) {
      sectCount++
      sectPayload += token.length
    } else {
      diffAbPayload += token.length
      diffAb.push(token)
    }
  }

  if (sectCount === 0 && uniquePatternA !== undefined && viewB !== undefined) {
    return disjointTokensRatio(
      diffAbPayload + diffAb.length - 1,
      uniquePatternA,
      viewB,
      tokensB,
      scoreCutoff,
    )
  }

  if (sectCount !== 0 && diffAb.length === 0) return 100

  const entriesB = sortedUniqueOf(tokensB)
  for (let i = 0; i < entriesB.keys.length; i++) {
    const token = entriesB.tokens[i]
    if (!tokensA.has(entriesB.keys[i], token)) {
      diffBaPayload += token.length
      diffBa.push(token)
    }
  }

  if (sectCount !== 0 && diffBa.length === 0) return 100

  const abLen = diffAbPayload + diffAb.length - 1
  const baLen = diffBaPayload + diffBa.length - 1
  const sectLen = sectCount === 0 ? 0 : sectPayload + sectCount - 1

  const sectAbLen = sectLen + (sectLen !== 0 ? 1 : 0) + abLen
  const sectBaLen = sectLen + (sectLen !== 0 ? 1 : 0) + baLen

  let sectAbRatio = 0
  let sectBaRatio = 0
  let cutoff = scoreCutoff
  if (sectLen !== 0) {
    sectAbRatio = normDistance(1 + abLen, sectLen + sectAbLen, scoreCutoff)
    sectBaRatio = normDistance(1 + baLen, sectLen + sectBaLen, scoreCutoff)
    if (sectAbRatio > cutoff) cutoff = sectAbRatio
    if (sectBaRatio > cutoff) cutoff = sectBaRatio
  }

  let result = 0
  const cutoffDistance = Math.ceil((sectAbLen + sectBaLen) * (1 - cutoff / 100))
  const lengthDiff = abLen > baLen ? abLen - baLen : baLen - abLen

  if (lengthDiff <= cutoffDistance) {
    const diffAbJoined = joinTokens(diffAb, abLen)
    const diffBaJoined = joinTokens(diffBa, baLen)
    const dist = indelDist(diffAbJoined, diffBaJoined, cutoffDistance)

    if (dist <= cutoffDistance) {
      result = normDistance(dist, sectAbLen + sectBaLen, cutoff)
    }
  }

  if (!sectLen) return result

  return Math.max(result, sectAbRatio, sectBaRatio)
}

export function tokenRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  preparedSortedPatternA?: () => PatternMask,
  preparedUniquePatternA?: () => PatternMask,
): number {
  if (scoreCutoff > 100) return 0

  const tokensViewA = viewA ?? tokenViewOf(a)
  const tokensViewB = viewB ?? tokenViewOf(b)

  const setScore = tokenSetRatioConverted(
    a,
    b,
    scoreCutoff,
    tokensViewA,
    tokensViewB,
    viewA === undefined ? undefined : preparedUniquePatternA,
  )

  if (setScore === 100) return 100

  return Math.max(
    setScore,
    tokenSortRatioConverted(
      a,
      b,
      Math.max(scoreCutoff, setScore),
      tokensViewA,
      tokensViewB,
      viewA === undefined ? undefined : preparedSortedPatternA,
    ),
  )
}

export function tokenRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  return tokenRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialTokenSortRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  const scoreCutoff = options.scoreCutoff ?? 0
  if (scoreCutoff > 100) return 0

  return partialRatioConverted(
    sortedOf(tokenViewOf(a)),
    sortedOf(tokenViewOf(b)),
    scoreCutoff,
  )
}

export function partialTokenSetRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  return partialTokenSetRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialTokenSetRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
): number {
  if (scoreCutoff > 100) return 0

  const tokensA = uniqueOf(viewA ?? tokenViewOf(a))
  const tokensB = uniqueOf(viewB ?? tokenViewOf(b))

  if (tokensA.size === 0 || tokensB.size === 0) return 0

  if (intersects(tokensA, tokensB)) return 100

  return partialRatioConverted(
    joinTokens(difference(tokensA, tokensB)),
    joinTokens(difference(tokensB, tokensA)),
    scoreCutoff,
  )
}

export function partialTokenRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = tokenPair(validateSequence(s1), validateSequence(s2))

  return partialTokenRatioConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialTokenRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
  viewA?: PreparedTokenChoice,
  viewB?: PreparedTokenChoice,
  preparedSortedPatternA?: () => PatternMask,
  preparedSortedCharSetA?: () => CharSet,
): number {
  if (scoreCutoff > 100) return 0

  const tokensViewA = viewA ?? tokenViewOf(a)
  const tokensViewB = viewB ?? tokenViewOf(b)
  const tokensA = uniqueOf(tokensViewA)
  const tokensB = uniqueOf(tokensViewB)

  if (intersects(tokensA, tokensB)) return 100

  const diffAb = difference(tokensA, tokensB)
  const diffBa = difference(tokensB, tokensA)

  const sortedA = sortedOf(tokensViewA)
  const sortedB = sortedOf(tokensViewB)
  const preparedApplies = viewA !== undefined && sortedA.length <= sortedB.length

  const result =
    partialAlignmentConverted(
      sortedA,
      sortedB,
      scoreCutoff,
      true,
      preparedApplies ? preparedSortedPatternA?.() : undefined,
      preparedApplies ? preparedSortedCharSetA?.() : undefined,
    )?.score ?? 0

  const splitA = splitOf(tokensViewA)
  const splitB = splitOf(tokensViewB)
  if (splitA.length === diffAb.length && splitB.length === diffBa.length) return result

  return Math.max(
    result,
    partialRatioConverted(
      joinTokens(diffAb),
      joinTokens(diffBa),
      Math.max(scoreCutoff, result),
    ),
  )
}
