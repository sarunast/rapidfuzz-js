// Ported from RapidFuzz tests/distance/test_Prefix.py
import { expect, it } from 'vitest'

import { defaultProcess } from '../../src/utils.js'
import { Prefix } from './scorers.js'

it('handles the basic cases', () => {
  expect(Prefix.distance('', '')).toBe(0)
  expect(Prefix.distance('test', 'test')).toBe(0)
  expect(Prefix.distance('aaaa', 'bbbb')).toBe(4)
})

it('applies score_cutoff correctly', () => {
  expect(Prefix.distance('abcd', 'abcee')).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { scoreCutoff: 4 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { scoreCutoff: 3 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { scoreCutoff: 2 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { scoreCutoff: 1 })).toBe(2)
  expect(Prefix.distance('abcd', 'abcee', { scoreCutoff: 0 })).toBe(1)
})

it('is case insensitive with the default processor', () => {
  expect(
    Prefix.distance('new york mets', 'new YORK mets', { processor: defaultProcess }),
  ).toBe(0)
})
