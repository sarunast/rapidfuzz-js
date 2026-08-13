// Not ported from RapidFuzz — upstream applies the Indel length ceiling inside
// its kernel, after both canonical forms exist, so it has no equivalent of the
// gate these pin.
//
// `tokenSortRatioConverted` measures the canonical length off the raw sequence
// and rejects on that ceiling before it splits, sorts or joins anything. The
// risk is a wrong rejection: a pair the ceiling refuses that the kernel would
// have scored comes back 0 rather than crashing, so only a value can catch it.
// Every expected score below was taken from rapidfuzz 3.14.5 on 2026-08-13.
import { describe, expect, it } from 'vitest'

import { convSequence } from '../../algorithms/shared/scorerSupport.js'
import { tokenViewOf } from './tokens.js'
import { tokenSortRatioConverted } from './tokenSort.js'
import { tokenSortRatio } from './tokenSortSimilarity.js'

describe('tokenSortRatio and the canonical length ceiling', () => {
  it('refuses a pair too differently sized to meet the cutoff', () => {
    expect(tokenSortRatio('aa bb', 'aa bb cc dd ee ff gg')).toBe(40)
    expect(tokenSortRatio('aa bb', 'aa bb cc dd ee ff gg', { scoreCutoff: 60 })).toBe(0)
  })

  it('does not refuse one that meets it exactly', () => {
    expect(tokenSortRatio('abc', 'abc def')).toBe(60)
    expect(tokenSortRatio('abc', 'abc def', { scoreCutoff: 60 })).toBe(60)
    expect(tokenSortRatio('abc', 'abc def', { scoreCutoff: 90 })).toBe(0)
  })

  // Both canonical lengths are 0, so the ceiling has no denominator. Two empty
  // inputs are identical, and a whitespace-only input tokenises to nothing.
  it('scores two empty canonical forms as identical, under a cutoff too', () => {
    expect(tokenSortRatio('', '')).toBe(100)
    expect(tokenSortRatio('', '', { scoreCutoff: 50 })).toBe(100)
    expect(tokenSortRatio('   ', '   ', { scoreCutoff: 50 })).toBe(100)
    expect(tokenSortRatio('', '   ', { scoreCutoff: 100 })).toBe(100)
  })

  // One side tokenises to nothing while the other does not: the ceiling is 0,
  // so any positive cutoff refuses it — which is the score anyway.
  it('scores an empty side against a non-empty one as 0', () => {
    expect(tokenSortRatio('   ', 'abc')).toBe(0)
    expect(tokenSortRatio('   ', 'abc', { scoreCutoff: 50 })).toBe(0)
  })

  // The canonical length is counted off whatever the sequence holds, so an
  // element that is not a code point takes the non-numeric arm of the scan.
  // Multi-character elements are not tokens: only a whitespace *element* splits
  // one, so `['alpha', 'beta']` is a single two-element token, and the pair
  // below shares one of those two elements.
  it('measures a sequence of arbitrary elements', () => {
    const left = ['alpha', 'beta']
    const right = ['beta', 'alpha']
    expect(tokenSortRatio(left, right)).toBe(50)
    expect(tokenSortRatio(left, right, { scoreCutoff: 90 })).toBe(0)
    expect(tokenSortRatio(left, right, { scoreCutoff: 50 })).toBe(50)
    expect(tokenSortRatio(['alpha'], right, { scoreCutoff: 90 })).toBe(0)
  })

  // A prepared query is held across candidates while the running cutoff climbs
  // from 0 — so the first candidate skips the gate and builds the sorted form,
  // and a later one measures the canonical length off that form instead of
  // rescanning the sequence. The two routes to the length must agree.
  it('agrees whether the sorted form was built before the length or after', () => {
    const query = convSequence('aa bb cc')
    const far = convSequence('aa bb cc dd ee ff gg hh')
    const near = convSequence('cc bb aa')

    // One view, reused: cutoff 0 first, so `sorted` exists before any length is
    // wanted, exactly as a best-match search reaches it.
    const held = tokenViewOf(query)
    const first = tokenSortRatioConverted(query, far, 0, held, tokenViewOf(far))
    const second = tokenSortRatioConverted(query, near, 50, held, tokenViewOf(near))

    // A third candidate reads the length the second one memoised.
    const third = tokenSortRatioConverted(query, far, 50, held, tokenViewOf(far))

    // Fresh views each time: the length comes off the scan instead.
    const freshFirst = tokenSortRatioConverted(query, far, 0)
    const freshSecond = tokenSortRatioConverted(query, near, 50)
    const freshThird = tokenSortRatioConverted(query, far, 50)

    expect(first).toBe(freshFirst)
    expect(second).toBe(freshSecond)
    expect(third).toBe(freshThird)
    expect(second).toBe(100)
  })
})
