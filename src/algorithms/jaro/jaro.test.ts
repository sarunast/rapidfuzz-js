// Ported from RapidFuzz tests/distance/test_Jaro.py
import { expect, it } from 'vitest'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { Jaro } from '../../../testing/scorers.js'
import { normalizeText as defaultProcess } from '../../core/normalize.js'
import { jaroSimilarity } from './implementation.js'

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

it('applies thresholds to the exact similarity, not just the coarse bound', () => {
  // The transpositions in "abcd"/"dcba" place the exact similarity (0.5) below
  // the coarse upper bound (0.66).
  expect(Jaro.similarity('abcd', 'dcba')).toBeCloseTo(0.5, 6)
  expect(Jaro.similarity('abcd', 'dcba', { threshold: 0.5 })).toBeCloseTo(0.5, 6)
  expect(Jaro.similarity('abcd', 'dcba', { threshold: 0.6 })).toBeUndefined()
})

it('reads thresholds on its natural normalized scale', () => {
  expect(Jaro.similarity('abcd', 'abce')).toBeCloseTo(0.833333, 5)
  expect(Jaro.similarity('abcd', 'abce', { threshold: 0.9 })).toBeUndefined()
})

it('compares normalized text case-insensitively', () => {
  expect(
    Jaro.similarity(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBeCloseTo(1, 6)
})

// Not ported — the flag buffers Jaro keeps between calls start at 32 words and
// grow, so a pattern past a thousand elements is the only thing that widens
// them. Twice, because the second call is the one that reuses what the first
// allocated.
it('scores a pattern wider than its retained flag buffers', () => {
  const a = 'abcdefghij'.repeat(200)
  const b = `${'abcdefghij'.repeat(199)}abcdefghix`

  expect(Jaro.similarity(a, a)).toBe(1)
  expect(Jaro.similarity(a, b)).toBeGreaterThan(0.99)
  expect(Jaro.similarity(a, b)).toBeLessThan(1)

  // Wider again, so the buffers grow rather than being reused, and then narrow
  // again, so the grown ones are reused rather than replaced. Differing
  // throughout, because a pair that differs in one place is trimmed to one
  // element before any buffer is sized.
  const wider = 'abcdefghij'.repeat(600)
  const widerEdited = [...wider].map((c, i) => (i % 11 === 0 ? 'z' : c)).join('')
  expect(Jaro.similarity(wider, widerEdited)).toBeLessThan(1)
  expect(Jaro.similarity(wider, widerEdited)).toBeGreaterThan(0.7)
  expect(Jaro.similarity(a, b)).toBeLessThan(1)
})

it('uses prepared prefix state in both word-width kernels', () => {
  const prepare = prepareScorerOf(jaroSimilarity)
  const oneWord = prepare('a'.repeat(32), {})
  expect(oneWord('a'.repeat(32), null)).toBe(1)

  const query = `abcde${'x'.repeat(40)}`
  const multiWord = prepare(query, {})
  expect(multiWord(`abcde${'y'.repeat(40)}`, 0)).toBeGreaterThan(0)
  expect(multiWord(`abcde${'y'.repeat(40)}`, 0.5)).toBe(0)

  const wordPrefix = prepare(`${'a'.repeat(32)}xxxxxxxx`, {})
  expect(wordPrefix(`${'a'.repeat(32)}yyyyyyyy`, 0)).toBeGreaterThan(0)
})
