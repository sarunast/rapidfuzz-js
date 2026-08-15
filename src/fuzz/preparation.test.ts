import { describe, expect, it } from 'vitest'

import { preparePattern } from '../algorithms/bitmask/pattern.js'
import {
  indelNormSimHeld,
  partialRatioAlignment_impl,
  partialRatioImpl,
} from './partialWindow.js'
import { prepareFuzz } from './preparation.js'
import { prepareRatio } from './ratio.js'
import {
  preparedTokenChoice,
  sortTokens,
  sortedOf,
  tokenChoicePreparer,
} from './token/tokens.js'
import { partialTokenRatioConverted } from './token/tokenSet.js'
import { prepareTokenSort } from './token/tokenSortRatio.js'

describe('fuzz preparation invariants', () => {
  it('covers fuzzRatio preparation cutoffs and bounds', () => {
    const preparation = prepareRatio()({})
    const empty = preparation.prepareQuery('')
    expect(empty(preparation.prepareChoice(''), null)).toBe(100)
    expect(empty(preparation.prepareChoice(''), 101)).toBe(0)

    const long = preparation.prepareQuery('a'.repeat(64))
    expect(long(preparation.prepareChoice('b'.repeat(64)), 100)).toBe(0)
    expect(long(preparation.prepareChoice('a'.repeat(64)), 70)).toBe(100)
    const short = preparation.prepareQuery('abc')
    expect(short(preparation.prepareChoice('abd'), 0)).toBeGreaterThan(0)
    expect(short(preparation.prepareChoice('abc'), 70)).toBe(100)
    expect(short(preparation.prepareChoice('axc'), 70)).toBe(0)
  })

  it('covers token-sort preparation', () => {
    const preparation = prepareTokenSort()({})
    const score = preparation.prepareQuery('b a')
    expect(score(preparation.prepareChoice('a b'), null)).toBe(100)
    expect(score(preparation.prepareChoice('a c'), 0)).toBeGreaterThan(0)
  })

  it('covers each composite prepared scorer and mixed representations', () => {
    const cases = [
      ['partialRatio', 'abc', [97, 98, 99, 100]],
      ['tokenSetRatio', 'new york', 'new jersey'],
      ['tokenRatio', 'new york', 'york new'],
      ['partialTokenSortRatio', 'alpha beta', 'beta alpha gamma'],
      ['partialTokenSetRatio', 'alpha beta', 'beta gamma'],
      ['partialTokenRatio', 'a a', 'b b'],
      ['weightedRatio', 'new york', 'new york city'],
    ] as const

    for (const [kind, query, choice] of cases) {
      const preparation = prepareFuzz(kind)({})
      const score = preparation.prepareQuery(query)
      expect(score(preparation.prepareChoice(choice), 0)).toBeGreaterThanOrEqual(0)
      expect(score(preparation.prepareChoice(choice), 101)).toBe(0)
    }

    const wPreparation = prepareFuzz('weightedRatio')({})
    const close = wPreparation.prepareQuery('abc')
    expect(close(wPreparation.prepareChoice('abd'), 0)).toBeGreaterThan(0)
    expect(close(wPreparation.prepareChoice('abc'), 70)).toBe(100)
    expect(close(wPreparation.prepareChoice('axc'), 70)).toBe(0)
    const bounded = wPreparation.prepareQuery('a'.repeat(64))
    expect(bounded(wPreparation.prepareChoice('b'.repeat(64)), 100)).toBe(0)
    const tokens = wPreparation.prepareQuery('a b')
    expect(tokens(wPreparation.prepareChoice('a c'), 0)).toBeGreaterThan(0)
    expect(tokens(wPreparation.prepareChoice('x'.repeat(20)), 70)).toBeGreaterThanOrEqual(
      0,
    )
    const wideRatio = wPreparation.prepareQuery('a')
    expect(
      wideRatio(wPreparation.prepareChoice('x'.repeat(20)), 0),
    ).toBeGreaterThanOrEqual(0)
    const longer = wPreparation.prepareQuery('alpha beta gamma delta')
    expect(longer(wPreparation.prepareChoice('alpha zeta'), 0)).toBeGreaterThanOrEqual(0)
    const empty = wPreparation.prepareQuery('')
    expect(empty(wPreparation.prepareChoice('x'), 0)).toBe(0)
    expect(close(wPreparation.prepareChoice(''), 0)).toBe(0)
  })

  it('covers partial alignment and token-difference second passes', () => {
    expect(indelNormSimHeld(preparePattern('', 0, 0), 0, '', 0, 0, 0)).toBe(1)
    // Two empty inputs are a perfect score, and a cutoff above 1 rejects even
    // that. No caller asks for one — each scales a percentage in first — but the
    // helper is exported, so it answers for itself rather than for them.
    expect(indelNormSimHeld(preparePattern('', 0, 0), 0, '', 0, 0, 1.1)).toBe(0)
    expect(partialRatioAlignment_impl('abc', 'zabc', { scoreCutoff: 101 })).toBeNull()
    expect(partialRatioImpl('abc', 'zabc', 1.1)).toEqual({
      score: 0,
      srcStart: 0,
      srcEnd: 3,
      destStart: 0,
      destEnd: 3,
    })
    expect(partialTokenRatioConverted('a a', 'b b', 0)).toBeGreaterThanOrEqual(0)

    const prepareChoice = tokenChoicePreparer()
    expect(() => preparedTokenChoice({})).toThrow(TypeError)
    expect(preparedTokenChoice(prepareChoice('a b')).sequence).toBeDefined()
    const nanChoice = preparedTokenChoice(prepareChoice([Number.NaN, ' ', Number.NaN]))
    expect(sortedOf(nanChoice)).toEqual([Number.NaN, 32, Number.NaN])
    expect(sortTokens([[Number.NaN], [Number.NaN]])).toEqual([[Number.NaN], [Number.NaN]])
    expect(sortTokens([[Number.NaN], [1]])).toEqual([[1], [Number.NaN]])
    expect(sortTokens([[1], [Number.NaN]])).toEqual([[1], [Number.NaN]])
  })
})
