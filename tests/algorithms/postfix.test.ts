// Ported from RapidFuzz tests/distance/test_Postfix.py
import { expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { Postfix } from '../support/scorers.js'

it('handles the basic cases', () => {
  expect(Postfix.distance('', '')).toBe(0)
  expect(Postfix.distance('test', 'test')).toBe(0)
  expect(Postfix.distance('aaaa', 'bbbb')).toBe(4)
})

it('applies score_cutoff correctly', () => {
  expect(Postfix.distance('abcd', 'eebcd')).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { scoreCutoff: 4 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { scoreCutoff: 3 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { scoreCutoff: 2 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { scoreCutoff: 1 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { scoreCutoff: 0 })).toBe(1)
})

it('is case insensitive with the default processor', () => {
  expect(
    Postfix.distance('new york mets', 'new YORK mets', { processor: defaultProcess }),
  ).toBe(0)
})
