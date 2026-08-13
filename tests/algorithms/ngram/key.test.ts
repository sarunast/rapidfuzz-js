import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import {
  canonicalRadix,
  feasibleRadices,
  packGram,
  unpackGram,
} from '../../../src/algorithms/shared/ngram/key.js'

describe('the radix ladder', () => {
  it('stays narrowest first, which is the order an index widens through', () => {
    // `NGramIndexBuilder` takes `feasibleRadices(gramSize)[0]` and re-keys
    // upward from it. Sorting this the other way to make `canonicalRadix`
    // cheaper would start every index at the widest rung it could ever need.
    expect(feasibleRadices(1)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(3)).toEqual([0x100, 0x1_0000])
    expect(feasibleRadices(6)).toEqual([0x100])
    expect(feasibleRadices(7)).toEqual([])
  })

  it('packs a prepared profile at the widest rung the depth allows', () => {
    // A profile has no corpus to re-key against, so it takes the widest rung up
    // front and falls back to a trie for anything that does not fit. Bigrams
    // reach the full code-point range, which is why an astral bigram packs and
    // an astral trigram does not.
    expect(canonicalRadix(1)).toBe(0x11_0000)
    expect(canonicalRadix(2)).toBe(0x11_0000)
    expect(canonicalRadix(3)).toBe(0x1_0000)
    expect(canonicalRadix(4)).toBe(0x100)
    expect(canonicalRadix(6)).toBe(0x100)
    expect(canonicalRadix(7)).toBeNull()
  })

  it('names the same rung the ladder would have chosen', () => {
    // `canonicalRadix` spells its answers out rather than filtering the ladder,
    // so nothing makes the two agree except this. A rung added to
    // `RADIX_LADDER` without a matching case there fails here.
    for (let gramSize = 1; gramSize <= 10; gramSize++) {
      const radices = feasibleRadices(gramSize)
      const widest = radices.length === 0 ? null : radices[radices.length - 1]
      expect(canonicalRadix(gramSize), `gramSize ${gramSize}`).toBe(widest)
    }
  })

  it('keeps every canonical key inside a safe integer', () => {
    for (const gramSize of [1, 2, 3, 4, 5, 6]) {
      const radix = canonicalRadix(gramSize)
      expect(radix, `gramSize ${gramSize}`).not.toBeNull()
      if (radix === null) continue
      expect(Math.pow(radix, gramSize), `gramSize ${gramSize}`).toBeLessThanOrEqual(
        Number.MAX_SAFE_INTEGER,
      )
    }
  })
})

describe('packing a gram', () => {
  it('round-trips every digit, most significant first', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 6 }), (gramSize) => {
        const radix = canonicalRadix(gramSize)
        if (radix === null) return
        return fc.assert(
          fc.property(
            fc.array(fc.integer({ min: 0, max: radix - 1 }), {
              minLength: gramSize,
              maxLength: gramSize,
            }),
            (digits) => {
              const key = packGram(digits, 0, gramSize, radix)
              expect(Number.isSafeInteger(key)).toBe(true)
              const decoded = new Array<number>(gramSize)
              unpackGram(key, gramSize, radix, decoded)
              expect(decoded).toEqual(digits)
            },
          ),
          { numRuns: 60 },
        )
      }),
      { numRuns: 24 },
    )
  })

  it('orders keys the way the elements read', () => {
    // The merge walk compares keys, so packing has to be monotone in the
    // elements: `['a', 'b']` must sort before `['a', 'c']`.
    const radix = canonicalRadix(2)
    expect(radix).toBe(0x11_0000)
    if (radix === null) return
    expect(packGram([97, 98], 0, 2, radix)).toBeLessThan(packGram([97, 99], 0, 2, radix))
    expect(packGram([97, 99], 0, 2, radix)).toBeLessThan(packGram([98, 0], 0, 2, radix))
  })
})
