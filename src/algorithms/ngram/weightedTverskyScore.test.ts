import { describe, expect, test } from 'vitest'

import { WEIGHTED_MASS_LIMIT, weightedTverskyScore } from './weightedTverskyScore.js'

describe('the weighted Tversky score', () => {
  test('is the ordinary ratio when nothing is extreme', () => {
    expect(weightedTverskyScore(4, 2, 0, 0.5, 0.5)).toBe(0.8)
    expect(weightedTverskyScore(4, 2, 0, 1, 1)).toBe(2 / 3)
    expect(weightedTverskyScore(4, 0, 6, 1, 0)).toBe(1)
    expect(weightedTverskyScore(0, 3, 3, 0.5, 0.5)).toBe(0)
  })

  test('prices a penalty a rounded mass would have lost', () => {
    // `massA` folds 1e16 + 1 back to 1e16, so a score derived from
    // `massA - shared` reports a perfect match with one token unmatched.
    expect(weightedTverskyScore(1e16, 1, 0, 1e16, 0)).toBe(0.5)
  })

  test('holds a subnormal shared mass against the largest weight', () => {
    // `shared / max(1, alpha)` underflows to 0, and the ordinary ratio would
    // then read 0 / 0 for a pair that shares everything.
    expect(weightedTverskyScore(Number.MIN_VALUE, 0, 0, Number.MAX_VALUE, 0)).toBe(1)
    expect(weightedTverskyScore(Number.MIN_VALUE, 0, 0, 0, Number.MAX_VALUE)).toBe(1)
  })

  test('weighs a subnormal penalty against a subnormal shared mass', () => {
    // Both operands survive `max(1, alpha, beta)` only as zero, and half the
    // mass is genuinely unmatched.
    expect(weightedTverskyScore(1e-300, 1e-300, 0, 1, Number.MAX_VALUE)).toBe(0.5)
  })

  test('keeps a penalty that the scaled weight alone would have flushed away', () => {
    // `alpha / scale` is subnormal, so its product with the penalty is 0 while
    // the numerator is not — the penalty is far below the shared mass either
    // way, so the answer is 1, reached without dividing 0 by 0.
    expect(weightedTverskyScore(1, 1e-300, 0, 1, Number.MAX_VALUE)).toBe(1)
  })

  test('reaches 0 rather than NaN when a penalty is beyond every score', () => {
    expect(weightedTverskyScore(1e-300, 1e300, 0, Number.MAX_VALUE, 0)).toBe(0)
    // No shared mass at all, with only the forgiven side carrying anything.
    expect(weightedTverskyScore(0, 1, 0, Number.MAX_VALUE, 0)).toBe(0)
    // Nothing shared and every operand subnormal: the scaled products vanish,
    // and a penalty over no shared mass is beyond every score rather than 0/0.
    const tiny = Number.MIN_VALUE
    expect(weightedTverskyScore(0, tiny, tiny, tiny, tiny)).toBe(0)
  })

  // The two the earlier fallback answered `0` for. It reached the penalty ratio
  // `alpha * firstOnly / shared`, which overflows for both while the score
  // itself is representable — the first as a subnormal, the second by a sum of
  // two `Number.MAX_VALUE`-scaled terms.
  test('recovers a score whose penalty ratio is past every representable number', () => {
    // `1e308 * 1e-300` is exactly `1e8`, so the ratio is computable here in the
    // ordinary way and says what the answer has to be.
    expect(weightedTverskyScore(1e-310, 1e-300, 0, 1e308, 0)).toBe(
      1e-310 / (1e-310 + 1e308 * 1e-300),
    )
    const tiny = Number.MIN_VALUE
    const huge = Number.MAX_VALUE
    expect(weightedTverskyScore(tiny, tiny, tiny, huge, huge)).toBe(
      tiny / (tiny + huge * tiny + huge * tiny),
    )
  })

  test('holds a weight 300 exponents below the largest one', () => {
    // `beta / max(1, alpha, beta)` is subnormal, so the fast path would price
    // `secondOnly` through a 26-bit factor and answer the eighth digit wrong.
    expect(weightedTverskyScore(1, 0, 4.4942328371557893e307, 1e16, 1e-300)).toBe(
      1 / (1 + 1e-300 * 4.4942328371557893e307),
    )
  })

  test('is symmetric under swapping both arguments and weights', () => {
    for (const [shared, first, second] of [
      [4, 2, 6],
      [1e16, 1, 3],
      [Number.MIN_VALUE, 0, 1e-300],
      [Number.MIN_VALUE, 1e16, Number.MIN_VALUE],
    ]) {
      for (const [alpha, beta] of [
        [0.5, 0.5],
        [1, 0],
        [2, 10],
        [Number.MAX_VALUE, 1],
        [1, Number.MIN_VALUE],
      ]) {
        expect(weightedTverskyScore(shared, first, second, alpha, beta)).toBe(
          weightedTverskyScore(shared, second, first, beta, alpha),
        )
      }
    }
  })

  test('stays inside 0..1 for every combination of extremes', () => {
    const masses = [
      0,
      Number.MIN_VALUE,
      1e-320,
      1e-310,
      1e-300,
      1,
      4,
      1e16,
      WEIGHTED_MASS_LIMIT,
    ]
    const weights = [0, Number.MIN_VALUE, 1e-300, 0.5, 1, 1e16, 1e308, Number.MAX_VALUE]
    for (const shared of masses) {
      for (const first of masses) {
        for (const second of masses) {
          for (const alpha of weights) {
            for (const beta of weights) {
              if (alpha === 0 && beta === 0) continue
              // Both sides carry mass whenever scoring is reached at all: a
              // side with none is answered by the zero-mass rule instead, and
              // `shared / (0 + 0)` is undefined for a reason.
              if (shared + first === 0 || shared + second === 0) continue
              const score = weightedTverskyScore(shared, first, second, alpha, beta)
              expect(score).toBeGreaterThanOrEqual(0)
              expect(score).toBeLessThanOrEqual(1)
            }
          }
        }
      }
    }
  })
})
