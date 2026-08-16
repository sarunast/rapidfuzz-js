import { describe, expect, it } from 'vitest'

import {
  exhaustive,
  indexOf,
  pairs,
  REPRESENTATION_SPECS,
} from '../../../../testing/invertedIndex.js'
import { feasibleRadices } from '../key.js'
import { createDiceIndexBuilder } from './dice.js'
import { repackKey } from './keys.js'

describe('the key scheme', () => {
  it('reaches every rung the gram size allows', () => {
    expect(feasibleRadices(1)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(2)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(3)).toEqual([0x100, 0x1_0000])
    expect(feasibleRadices(6)).toEqual([0x100])
    // Seven bytes is 2^56, past a safe integer, so no packed rung survives and
    // the joined-string scheme is the only one left.
    expect(feasibleRadices(7)).toEqual([])
  })

  it('widens from a byte to BMP, and again to strings, inside one choice', () => {
    // A byte-radix index, then a lone surrogate that needs BMP, then an astral
    // character that no trigram radix holds at all.
    const choices = ['abc', '\ud800bc', '😀bc']
    for (const spec of REPRESENTATION_SPECS) {
      const index = indexOf(spec, 3, choices)
      for (const query of choices) {
        expect(pairs(index.select(query, 0.99, 1))).toEqual(
          exhaustive(spec, 3, choices, query, 0.99, 1),
        )
      }
    }
  })

  it('keeps a joined-string index exact', () => {
    // Gram size 3 over astral text has no feasible packed radix at all.
    const choices = ['😀😁😂', '😀😁😃', '😀😁😂😄']
    for (const spec of REPRESENTATION_SPECS) {
      const index = indexOf(spec, 3, choices)
      for (const query of [...choices, '😀😁']) {
        expect(pairs(index.select(query, null, null))).toEqual(
          exhaustive(spec, 3, choices, query, null, null),
        )
      }
    }
  })

  it('drops to joined strings for a negative element', () => {
    // An array choice may hold any integer, and positional packing has no room
    // below zero — so a negative element takes the whole index to strings.
    const builder = createDiceIndexBuilder(2)
    for (const choice of [
      [1, 2, 3],
      [-1, 2, 3],
      [1, 2, 3],
    ])
      builder.add(choice)
    const index = builder.seal()
    expect(pairs(index.select([1, 2, 3], 0.99, 3))).toEqual([
      { id: 0, score: 1 },
      { id: 2, score: 1 },
    ])
    expect(pairs(index.select([-1, 2, 3], 0.99, 3))).toEqual([{ id: 1, score: 1 }])
  })

  it('starts on joined strings when no packed rung can hold the gram', () => {
    // Seven bytes is 2^56, past a safe integer, so the ladder is empty and the
    // index is string-keyed from the first choice.
    const builder = createDiceIndexBuilder(7)
    for (const choice of ['abcdefgh', 'abcdefgi']) builder.add(choice)
    const index = builder.seal()
    expect(pairs(index.select('abcdefgh', 0.5, 2))).toEqual(
      exhaustive({ metric: 'dice' }, 7, ['abcdefgh', 'abcdefgi'], 'abcdefgh', 0.5, 2),
    )
  })

  it('leaves an already-joined key alone when the scheme widens', () => {
    expect(repackKey('1,2,3', 0x100, null, 3)).toBe('1,2,3')
    expect(repackKey(0x616263, 0x100, 0x1_0000, 3)).toBe(
      0x61 * 0x1_0000 * 0x1_0000 + 0x62 * 0x1_0000 + 0x63,
    )
    expect(repackKey(0x616263, 0x100, null, 3)).toBe('97,98,99')
  })

  it('counts a query gram no packed index could hold', () => {
    // The index is byte-keyed; the query's astral grams cannot appear in it, and
    // still have to count toward the query's own gram count and norm.
    const choices = ['abcd', 'abce']
    for (const spec of REPRESENTATION_SPECS) {
      const index = indexOf(spec, 2, choices)
      expect(pairs(index.select('ab😀cd', null, null))).toEqual(
        exhaustive(spec, 2, choices, 'ab😀cd', null, null),
      )
    }
  })
})
