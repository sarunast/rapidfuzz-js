import { describe, expect, it } from 'vitest'

import { preparePattern } from '../../src/algorithms/shared/bitmask/pattern.js'
import { PREPARE_CHOICE } from '../../src/algorithms/shared/scorerSupport.js'
import {
  indelNormSimHeld,
  partialRatioAlignment,
  partialRatioImpl,
} from '../../src/fuzz/internal/partialWindow.js'
import { prepareFuzz } from '../../src/fuzz/internal/prepared.js'
import {
  preparedTokenChoice,
  sortTokens,
  sortedOf,
  tokenChoicePreparer,
} from '../../src/fuzz/internal/tokens.js'
import { partialTokenRatioConverted } from '../../src/fuzz/internal/tokenSet.js'
import { prepareSimilarity } from '../../src/fuzz/similarity.js'
import { prepareTokenSort } from '../../src/fuzz/tokenSort.js'

describe('fuzz preparation invariants', () => {
  it('covers ratio preparation cutoffs and bounds', () => {
    const factory = prepareSimilarity()
    const empty = factory('', {})
    expect(empty(factory[PREPARE_CHOICE](''), null)).toBe(100)
    expect(empty(factory[PREPARE_CHOICE](''), 101)).toBe(0)

    const long = factory('a'.repeat(64), {})
    expect(long(factory[PREPARE_CHOICE]('b'.repeat(64)), 100)).toBe(0)
    expect(long(factory[PREPARE_CHOICE]('a'.repeat(64)), 70)).toBe(100)
    const short = factory('abc', {})
    expect(short(factory[PREPARE_CHOICE]('abd'), 0)).toBeGreaterThan(0)
    expect(short(factory[PREPARE_CHOICE]('abc'), 70)).toBe(100)
    expect(short(factory[PREPARE_CHOICE]('axc'), 70)).toBe(0)
  })

  it('covers token-sort preparation', () => {
    const factory = prepareTokenSort()
    const score = factory('b a', {})
    expect(score(factory[PREPARE_CHOICE]('a b'), null)).toBe(100)
    expect(score(factory[PREPARE_CHOICE]('a c'), 0)).toBeGreaterThan(0)
  })

  it('covers each composite prepared scorer and mixed representations', () => {
    const cases = [
      ['partialRatio', 'abc', [97, 98, 99, 100]],
      ['tokenSetRatio', 'new york', 'new jersey'],
      ['tokenRatio', 'new york', 'york new'],
      ['partialTokenSortRatio', 'alpha beta', 'beta alpha gamma'],
      ['partialTokenSetRatio', 'alpha beta', 'beta gamma'],
      ['partialTokenRatio', 'a a', 'b b'],
      ['wRatio', 'new york', 'new york city'],
    ] as const

    for (const [kind, query, choice] of cases) {
      const factory = prepareFuzz(kind)
      const score = factory(query, {})
      expect(score(factory[PREPARE_CHOICE](choice), 0)).toBeGreaterThanOrEqual(0)
      expect(score(factory[PREPARE_CHOICE](choice), 101)).toBe(0)
    }

    const wFactory = prepareFuzz('wRatio')
    const close = wFactory('abc', {})
    expect(close(wFactory[PREPARE_CHOICE]('abd'), 0)).toBeGreaterThan(0)
    expect(close(wFactory[PREPARE_CHOICE]('abc'), 70)).toBe(100)
    expect(close(wFactory[PREPARE_CHOICE]('axc'), 70)).toBe(0)
    const bounded = wFactory('a'.repeat(64), {})
    expect(bounded(wFactory[PREPARE_CHOICE]('b'.repeat(64)), 100)).toBe(0)
    const tokens = wFactory('a b', {})
    expect(tokens(wFactory[PREPARE_CHOICE]('a c'), 0)).toBeGreaterThan(0)
    expect(tokens(wFactory[PREPARE_CHOICE]('x'.repeat(20)), 70)).toBeGreaterThanOrEqual(0)
    const wideRatio = wFactory('a', {})
    expect(wideRatio(wFactory[PREPARE_CHOICE]('x'.repeat(20)), 0)).toBeGreaterThanOrEqual(
      0,
    )
    const longer = wFactory('alpha beta gamma delta', {})
    expect(longer(wFactory[PREPARE_CHOICE]('alpha zeta'), 0)).toBeGreaterThanOrEqual(0)
    const empty = wFactory('', {})
    expect(empty(wFactory[PREPARE_CHOICE]('x'), 0)).toBe(0)
    expect(close(wFactory[PREPARE_CHOICE](''), 0)).toBe(0)
  })

  it('covers partial alignment and token-difference second passes', () => {
    expect(indelNormSimHeld(preparePattern('', 0, 0), 0, '', 0, 0, 0)).toBe(1)
    // Two empty inputs are a perfect score, and a cutoff above 1 rejects even
    // that. No caller asks for one — each scales a percentage in first — but the
    // helper is exported, so it answers for itself rather than for them.
    expect(indelNormSimHeld(preparePattern('', 0, 0), 0, '', 0, 0, 1.1)).toBe(0)
    expect(partialRatioAlignment('abc', 'zabc', { scoreCutoff: 101 })).toBeNull()
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
