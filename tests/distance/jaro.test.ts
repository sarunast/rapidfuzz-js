// Ported from RapidFuzz tests/distance/test_Jaro.py
import { expect, it } from 'vitest'

import { defaultProcess } from '../../src/utils.js'
import { Jaro } from './scorers.js'

it('handles sequences of numbers', () => {
  expect(Jaro.similarity([0, -1], [0, -2])).toBeCloseTo(0.666666, 5)
})

it('handles the edge case lengths found by fuzzing', () => {
  expect(Jaro.similarity('', '')).toBeCloseTo(1, 6)
  expect(Jaro.similarity('0', '0')).toBeCloseTo(1, 6)
  expect(Jaro.similarity('00', '00')).toBeCloseTo(1, 6)
  expect(Jaro.similarity('0', '00')).toBeCloseTo(0.833333, 5)

  expect(Jaro.similarity('0'.repeat(65), '0'.repeat(65))).toBeCloseTo(1, 6)
  expect(Jaro.similarity('0'.repeat(64), '0'.repeat(65))).toBeCloseTo(0.994872, 5)
  expect(Jaro.similarity('0'.repeat(63), '0'.repeat(65))).toBeCloseTo(0.989744, 5)

  expect(Jaro.similarity('000000001', '0000010')).toBeCloseTo(0.878307, 5)

  expect(
    Jaro.similarity('01234567', '0'.repeat(170) + '7654321' + '0'.repeat(200)),
  ).toBeCloseTo(0.54874, 5)

  expect(
    Jaro.similarity(
      '10000000000000000000000000000000000000000000000000000000000000020',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.979487, 5)

  expect(
    Jaro.similarity(
      '0000000000000000000000000000000000000000000000000000000000000000000000000000001',
      '00000000000000100000000000000000000000010000000000000000000000000',
    ),
  ).toBeCloseTo(0.922233, 5)

  expect(
    Jaro.similarity(
      '010000000000000000000000000000000000000000000000000000000000000000' +
        '00000000000000000000000000000000000000000000000000000000000000',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.8359375, 6)
})

it('applies score_cutoff to the exact similarity, not just the coarse bound', () => {
  // The transpositions in "abcd"/"dcba" place the exact similarity (0.5) below
  // the coarse upper bound (0.66).
  expect(Jaro.similarity('abcd', 'dcba')).toBeCloseTo(0.5, 6)
  expect(Jaro.similarity('abcd', 'dcba', { scoreCutoff: 0.5 })).toBeCloseTo(0.5, 6)
  expect(Jaro.similarity('abcd', 'dcba', { scoreCutoff: 0.6 })).toBe(0)
  expect(Jaro.normalizedSimilarity('abcd', 'dcba', { scoreCutoff: 0.6 })).toBe(0)
})

// Not ported — upstream has no test for it, but its `Jaro.distance` and
// `Jaro.similarity` take `score_cutoff` as a `double` in `[0, 1]`, exactly as
// the `normalized_*` pair does, because Jaro has no raw score for a cutoff to
// count elements of. The values below are rapidfuzz 3.14.5's.
//
// The distinction is visible in three places: a fraction is a real cutoff
// rather than something truncated to `0`, a rejected pair reports the worst
// score rather than `scoreCutoff + 1`, and a cutoff outside `[0, 1]` is
// refused rather than quietly rejecting everything.
it('reads scoreCutoff as a normalised one, distance included', () => {
  expect(Jaro.similarity('abcd', 'abce')).toBeCloseTo(0.833333, 5)
  expect(Jaro.similarity('abcd', 'abce', { scoreCutoff: 0.9 })).toBe(0)
  expect(Jaro.distance('abcd', 'abce', { scoreCutoff: 0.1 })).toBe(1)
  expect(Jaro.distance('abcd', 'abce', { scoreCutoff: 0.5 })).toBeCloseTo(0.166666, 5)
  expect(() => Jaro.similarity('abcd', 'abce', { scoreCutoff: 1.5 })).toThrow(RangeError)
  expect(() => Jaro.distance('abcd', 'abce', { scoreCutoff: -1 })).toThrow(RangeError)
})

it('is case insensitive with the default processor', () => {
  expect(
    Jaro.similarity('new york mets', 'new YORK mets', { processor: defaultProcess }),
  ).toBeCloseTo(1, 6)
})
