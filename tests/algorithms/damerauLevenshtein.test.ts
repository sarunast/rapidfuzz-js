// Ported from RapidFuzz tests/distance/test_DamerauLevenshtein.py
import { describe, expect, it } from 'vitest'

import { resetDamerauScratch } from '../../src/algorithms/damerauLevenshtein/implementation.js'
import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { DamerauLevenshtein } from '../support/scorers.js'

const CASES: ReadonlyArray<readonly [string, string, number]> = [
  ['test', 'text', 1],
  ['test', 'tset', 1],
  ['test', 'qwy', 4],
  ['test', 'testit', 2],
  ['test', 'tesst', 1],
  ['test', 'tet', 1],
  ['cat', 'hat', 1],
  ['Niall', 'Neil', 3],
  ['aluminum', 'Catalan', 7],
  ['ATCG', 'TAGC', 2],
  ['ab', 'ba', 1],
  ['ab', 'cde', 3],
  ['ab', 'ac', 1],
  ['ab', 'bc', 2],
  ['ca', 'abc', 2],
]

describe('distance', () => {
  for (const [left, right, distance] of CASES) {
    it(`${left} -> ${right} is ${distance}`, () => {
      expect(DamerauLevenshtein.distance(left, right)).toBe(distance)
    })
  }
})

it('compares normalized text case-insensitively', () => {
  expect(
    DamerauLevenshtein.distance(
      defaultProcess('new york mets'),
      defaultProcess('new YORK mets'),
    ),
  ).toBe(0)
})

it('grows and reuses wide retained rows', () => {
  resetDamerauScratch()
  const long = new Array<string>(0x8000).fill('a')
  long[0] = 'b'
  expect(DamerauLevenshtein.distance(long, new Array(65).fill('z'))).toBe(long.length)
  expect(DamerauLevenshtein.distance(long, new Array(129).fill('z'))).toBe(long.length)
  expect(DamerauLevenshtein.distance(long, new Array(100).fill('z'))).toBe(long.length)
})

it('rejects a generic sequence after the full bounded matrix', () => {
  expect(DamerauLevenshtein.distance([1, 2], [3, 4], { threshold: 1 })).toBeUndefined()
})

it('applies normalized similarity thresholds to direct scoring', () => {
  expect(DamerauLevenshtein.normalizedSimilarity('abcd', 'abce')).toBe(0.75)
  expect(
    DamerauLevenshtein.normalizedSimilarity('abcd', 'abce', { threshold: 0.8 }),
  ).toBeUndefined()
})
