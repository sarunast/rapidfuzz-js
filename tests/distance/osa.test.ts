// Ported from RapidFuzz tests/distance/test_OSA.py
import { expect, it } from 'vitest'

import { osaDistance } from '../../src/algorithms/osa/implementation.js'
import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { OSA } from './scorers.js'

it('treats two empty strings as a perfect match', () => {
  expect(OSA.distance('', '')).toBe(0)
})

it('interprets strings and sequences the same way', () => {
  expect(OSA.distance('aaaa', 'aaaa')).toBe(0)
  expect(OSA.distance('aaaa', ['a', 'a', 'a', 'a'])).toBe(0)
  expect(osaDistance([0, -1], [0, -2])).toBe(1)
})

it('can express a word error rate over token sequences', () => {
  expect(OSA.distance(['aaaaa', 'bbbb'], ['aaaaa', 'bbbb'])).toBe(0)
  expect(OSA.distance(['aaaaa', 'bbbb'], ['aaaaa', 'cccc'])).toBe(1)
})

it('restricts each substring to a single edit', () => {
  expect(OSA.distance('CA', 'ABC')).toBe(3)
  expect(OSA.distance('CA', 'AC')).toBe(1)
  expect(
    OSA.distance(
      'a'.repeat(65) + 'CA' + 'a'.repeat(65),
      'b' + 'a'.repeat(64) + 'AC' + 'a'.repeat(64) + 'b',
    ),
  ).toBe(3)
})

it('trims long common affixes without changing OSA transpositions', () => {
  const prefix = 'a'.repeat(2048)
  const suffix = 'z'.repeat(2048)
  expect(OSA.distance(`${prefix}CA${suffix}`, `${prefix}AC${suffix}`)).toBe(1)
  expect(OSA.distance(`${prefix}CA${suffix}`, `${prefix}ABC${suffix}`)).toBe(3)
})

it('handles unicode', () => {
  const s1 = 'ÁÄ'
  expect(OSA.distance(s1, 'ABCD')).toBe(4)
  expect(OSA.distance(s1, s1)).toBe(0)
})

it('is case insensitive with the default processor', () => {
  expect(
    OSA.distance('new york mets', 'new YORK mets', { processor: defaultProcess }),
  ).toBe(0)
})
