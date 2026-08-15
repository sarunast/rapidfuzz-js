// Ported from RapidFuzz tests/distance/test_Prefix.py
import { expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '#core/normalize.js'

import { Prefix } from '../../../testing/scorers.js'

it('handles the basic cases', () => {
  expect(Prefix.distance('', '')).toBe(0)
  expect(Prefix.distance('test', 'test')).toBe(0)
  expect(Prefix.distance('aaaa', 'bbbb')).toBe(4)
})

it('applies native distance thresholds', () => {
  expect(Prefix.distance('abcd', 'abcee')).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { threshold: 4 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { threshold: 3 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { threshold: 2 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { threshold: 1 })).toBeUndefined()
  expect(Prefix.distance('abcd', 'abcee', { threshold: 0 })).toBeUndefined()
})

it('compares normalized text case-insensitively', () => {
  expect(
    Prefix.distance(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(0)
})
