// Ported from RapidFuzz tests/distance/test_DamerauLevenshtein.py
import { describe, expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { DamerauLevenshtein } from './scorers.js'

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

it('is case insensitive with the default processor', () => {
  expect(
    DamerauLevenshtein.distance('new york mets', 'new YORK mets', {
      processor: defaultProcess,
    }),
  ).toBe(0)
})
