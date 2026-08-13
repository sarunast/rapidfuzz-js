// Not ported from RapidFuzz — upstream computes the two section ratios after
// the difference comparison, so it has no equivalent of the shortcut these pin.
//
// `tokenSetRatioConverted` scores the sections first and lets them raise the
// cutoff the difference comparison has to clear. Whenever the joined difference
// lengths are further apart than that cutoff allows, no difference score could
// have been reported, so both differences are left unsorted and unjoined and the
// kernel is never called.
//
// The risk the shortcut carries is a wrong-but-plausible score rather than a
// crash: a bound off by one returns the section ratio where a difference score
// should have won. Every expected value below was taken from rapidfuzz 3.14.5 on
// 2026-08-13, not from this implementation.
import { describe, expect, it } from 'vitest'

import { wRatio } from '../weightedSimilarity.js'
import { tokenSetRatio } from './tokenSetSimilarity.js'
import { tokenRatio } from './tokenSimilarity.js'

describe('tokenSetRatio when the differences cannot be compared', () => {
  // Shared section of 11 elements against differences of 1 and 20, so the
  // section ratio is 91.66… and the difference comparison would need to beat it
  // with a distance of at most 4 — against a length difference of 19.
  const short = 'aa bb cc dd x'
  const long = 'aa bb cc dd yyyyyyyyyyyyyyyyyyyy'

  it('reports the section ratio', () => {
    expect(tokenSetRatio(short, long)).toBeCloseTo(91.66666666666667, 12)
  })

  it('reports it whichever way round the sides arrive', () => {
    expect(tokenSetRatio(long, short)).toBeCloseTo(tokenSetRatio(short, long), 12)
  })

  it('is unchanged by a cutoff below the answer', () => {
    expect(tokenSetRatio(short, long, { scoreCutoff: 50 })).toBeCloseTo(
      91.66666666666667,
      12,
    )
  })

  it('still answers 0 to a cutoff above the answer', () => {
    expect(tokenSetRatio(short, long, { scoreCutoff: 95 })).toBe(0)
  })

  it('carries through the scorers built on it', () => {
    expect(tokenRatio(short, long)).toBeCloseTo(91.66666666666667, 12)
    expect(wRatio(short, long)).toBeCloseTo(86.4, 12)
  })

  // No shared section, so nothing raises the cutoff — but a caller-supplied one
  // still closes the gate. RapidFuzz scores this pair 33.33…, which a cutoff of
  // 90 rejects either way.
  it('answers 0 when only the caller cutoff closes the gate', () => {
    expect(tokenSetRatio('a b c', 'c d eeeeeeeeeeeeeeeeeeee')).toBeCloseTo(
      33.33333333333333,
      12,
    )
    expect(tokenSetRatio('a b c', 'c d eeeeeeeeeeeeeeeeeeee', { scoreCutoff: 90 })).toBe(
      0,
    )
  })
})
