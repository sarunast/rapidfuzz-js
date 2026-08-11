// Ported from RapidFuzz tests/test_utils.py
import { expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'

it('preprocesses mixed unicode strings', () => {
  const mixed: ReadonlyArray<readonly [string, string]> = [
    [
      'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
      'lorem ipsum is simply dummy text of the printing and typesetting industry',
    ],
    ["C'est la vie", 'c est la vie'],
    ['Ça va?', 'ça va'],
    ['Cães danados', 'cães danados'],
    ['¬Camarões assados', 'camarões assados'],
    ['a¬ሴ€耀', 'a ሴ 耀'],
    ['Á', 'á'],
  ]

  for (const [input, expected] of mixed) {
    expect(defaultProcess(input)).toBe(expected)
  }
})

it('rejects non-string input', () => {
  expect(() => defaultProcess(['a', 'b'])).toThrow('normalizeText expects a string')
})
