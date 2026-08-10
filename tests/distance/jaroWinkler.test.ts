// Ported from RapidFuzz tests/distance/test_JaroWinkler.py
import { expect, it } from 'vitest'

import { defaultProcess } from '../../src/utils.js'
import { JaroWinkler } from './scorers.js'

it('handles sequences of numbers', () => {
  expect(JaroWinkler.similarity([0, -1], [0, -2])).toBeCloseTo(0.666666, 5)
})

it('clamps to 1.0 with a large prefix weight', () => {
  expect(
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 0.5 }),
  ).toBeCloseTo(1, 6)
  expect(
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 1 }),
  ).toBeCloseTo(1, 6)
})

// Not ported — see the equivalent in `jaro.test.ts` for why `distance` and
// `similarity` read `scoreCutoff` as a normalised cutoff here. Values from
// rapidfuzz 3.14.5.
it('reads scoreCutoff as a normalised one, distance included', () => {
  expect(JaroWinkler.similarity('abcd', 'abce')).toBeCloseTo(0.883333, 5)
  expect(JaroWinkler.similarity('abcd', 'abce', { scoreCutoff: 0.95 })).toBe(0)
  expect(JaroWinkler.distance('abcd', 'abce', { scoreCutoff: 0.1 })).toBe(1)
  expect(JaroWinkler.distance('abcd', 'abce', { scoreCutoff: 0.5 })).toBeCloseTo(
    0.116666,
    5,
  )
  expect(() => JaroWinkler.similarity('abcd', 'abce', { scoreCutoff: 1.5 })).toThrow(
    RangeError,
  )
})

it('rejects an out-of-range prefix weight', () => {
  expect(() =>
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: -0.1 }),
  ).toThrow('prefix_weight has to be in the range 0.0 - 1.0')

  expect(() =>
    JaroWinkler.similarity('milyarder', 'milyarderlik', { prefixWeight: 1.1 }),
  ).toThrow('prefix_weight has to be in the range 0.0 - 1.0')
})

it('handles the edge case lengths found by fuzzing', () => {
  expect(JaroWinkler.similarity('', '')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0', '0')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('00', '00')).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0', '00')).toBeCloseTo(0.85, 6)

  expect(JaroWinkler.similarity('0'.repeat(65), '0'.repeat(65))).toBeCloseTo(1, 6)
  expect(JaroWinkler.similarity('0'.repeat(64), '0'.repeat(65))).toBeCloseTo(0.996923, 5)
  expect(JaroWinkler.similarity('0'.repeat(63), '0'.repeat(65))).toBeCloseTo(0.993846, 5)

  expect(JaroWinkler.similarity('000000001', '0000010')).toBeCloseTo(0.926984, 5)

  expect(
    JaroWinkler.similarity(
      '10000000000000000000000000000000000000000000000000000000000000020',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.979487, 5)

  expect(
    JaroWinkler.similarity(
      '0000000000000000000000000000000000000000000000000000000000000000000000000000001',
      '00000000000000100000000000000000000000010000000000000000000000000',
    ),
  ).toBeCloseTo(0.95334, 5)

  expect(
    JaroWinkler.similarity(
      '010000000000000000000000000000000000000000000000000000000000000000' +
        '00000000000000000000000000000000000000000000000000000000000000',
      '00000000000000000000000000000000000000000000000000000000000000000',
    ),
  ).toBeCloseTo(0.852344, 6)
})

it('is case insensitive with the default processor', () => {
  expect(
    JaroWinkler.similarity('new york mets', 'new YORK mets', {
      processor: defaultProcess,
    }),
  ).toBeCloseTo(1, 6)
})
