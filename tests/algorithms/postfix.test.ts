// Ported from RapidFuzz tests/distance/test_Postfix.py
import { expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { Postfix } from '../support/scorers.js'

it('handles the basic cases', () => {
  expect(Postfix.distance('', '')).toBe(0)
  expect(Postfix.distance('test', 'test')).toBe(0)
  expect(Postfix.distance('aaaa', 'bbbb')).toBe(4)
})

it('applies native distance thresholds', () => {
  expect(Postfix.distance('abcd', 'eebcd')).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { threshold: 4 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { threshold: 3 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { threshold: 2 })).toBe(2)
  expect(Postfix.distance('abcd', 'eebcd', { threshold: 1 })).toBeUndefined()
  expect(Postfix.distance('abcd', 'eebcd', { threshold: 0 })).toBeUndefined()
})

it('compares normalized text case-insensitively', () => {
  expect(
    Postfix.distance(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(0)
})
