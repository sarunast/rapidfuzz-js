// Not ported from RapidFuzz — it guards an invariant this port has and upstream
// does not, because upstream never hands a tokenisation back to a caller.
//
// `sortedOf` sorts `split` in place instead of copying the outer array first.
// That is only sound while `split` carries no ordering contract: it is the token
// multiset, not the original textual order. Sorting it changes what a *later*
// `uniqueOf` sees — which instance of each equal token is retained, and the
// insertion order of the maps behind `UniqueTokenSet` — so the accessors must
// give one answer whichever order they are driven in.
//
// These reach into `tokens.js` directly on purpose. The invariant is internal, and
// asserting it through the public scorers alone would only test the access order
// those scorers happen to use today.
import { describe, expect, it } from 'vitest'

import { convSequence } from '../../core/sequence.js'
import { fuzzTokenRatio } from './tokenRatio.js'
import {
  difference,
  intersects,
  sortedOf,
  splitOf,
  tokenViewOf,
  uniqueOf,
  type UniqueTokenSet,
} from './tokens.js'
import { fuzzTokenSetRatio } from './tokenSetRatio.js'
import { fuzzTokenSortRatio } from './tokenSortRatio.js'

/** Everything observable about a token set, in an order-free form. */
function snapshot(set: UniqueTokenSet): { size: number; tokens: string[] } {
  const tokens: string[] = []
  for (const [, token] of set.packed) tokens.push(JSON.stringify(token))
  for (const [, bucket] of set.mixed) {
    for (const token of bucket) tokens.push(JSON.stringify(token))
  }
  return { size: set.size, tokens: tokens.sort() }
}

const texts = [
  'delta alpha charlie bravo',
  'alpha alpha beta',
  'the quick brown fox jumps over the lazy dog',
  'b a',
  'a',
  '',
  '   ',
  'zzz aaa zzz aaa mmm',
  'café naïve café',
  '\u{1F600} \u{1F601} \u{1F600}',
]

describe('split carries no ordering contract', () => {
  it.each(texts)('gives one answer whichever accessor runs first for %j', (text) => {
    const sequence = convSequence(text)

    // unique first, so it sees the split as `splitSequence` built it.
    const uniqueFirst = tokenViewOf(sequence)
    const uniqueBefore = snapshot(uniqueOf(uniqueFirst))
    const sortedAfter = sortedOf(uniqueFirst)
    // Asking again after the in-place sort must not change the answer.
    const uniqueAfter = snapshot(uniqueOf(uniqueFirst))

    // sorted first, so `uniqueOf` sees an already-sorted split.
    const sortedFirst = tokenViewOf(sequence)
    const sortedBefore = sortedOf(sortedFirst)
    const uniqueFromSorted = snapshot(uniqueOf(sortedFirst))

    expect(sortedAfter).toEqual(sortedBefore)
    expect(uniqueAfter).toEqual(uniqueBefore)
    expect(uniqueFromSorted).toEqual(uniqueBefore)
    expect(splitOf(uniqueFirst).length).toBe(splitOf(sortedFirst).length)
  })

  it('leaves difference and intersects order-free', () => {
    for (const left of texts) {
      for (const right of texts) {
        const a1 = tokenViewOf(convSequence(left))
        const b1 = tokenViewOf(convSequence(right))
        const plain = difference(uniqueOf(a1), uniqueOf(b1))
          .map((token) => JSON.stringify(token))
          .sort()
        const meets = intersects(uniqueOf(a1), uniqueOf(b1))

        // The same pair, but with both splits sorted in place beforehand.
        const a2 = tokenViewOf(convSequence(left))
        const b2 = tokenViewOf(convSequence(right))
        sortedOf(a2)
        sortedOf(b2)
        const afterSort = difference(uniqueOf(a2), uniqueOf(b2))
          .map((token) => JSON.stringify(token))
          .sort()

        expect(afterSort).toEqual(plain)
        expect(intersects(uniqueOf(a2), uniqueOf(b2))).toBe(meets)
      }
    }
  })
})

// The scorers reach the accessors in different orders — `tokenSetRatio` asks
// only for the unique set, `tokenSortRatio` only for the sorted form, and
// `tokenRatio` for the set and then, on some inputs, the sorted form. Scoring a
// pair every way round would expose an order dependence the unit checks missed.
describe('token scorers agree regardless of which forms they build', () => {
  it.each(texts)('scores %j consistently against every other text', (text) => {
    for (const other of texts) {
      const set = fuzzTokenSetRatio(text, other)
      const sort = fuzzTokenSortRatio(text, other)
      const both = fuzzTokenRatio(text, other)

      // `tokenRatio` is defined as the larger of the two, so this is the
      // relationship that an order-dependent split would break.
      expect(both).toBeCloseTo(Math.max(set, sort), 9)
    }
  })
})

// `difference` and `UniqueTokenSet.has` are exported, and the production
// callers reach them only after `intersects` has answered false — so the
// shared-token half of each is unreachable *through a scorer* and reachable
// through the export. The test above already drives the packed half that way;
// these drive the mixed half, and `has` for a packed key, which only a caller
// walking `packed` itself would ask about.
describe('the token set accessors answer for a shared token', () => {
  const shared = { tag: 'shared' }
  const onlyLeft = { tag: 'left' }
  const onlyRight = { tag: 'right' }

  /** One token per element, so a set is the bag of elements it was built from. */
  function setOf(elements: readonly unknown[]): UniqueTokenSet {
    const sequence: unknown[] = []
    for (const element of elements) {
      if (sequence.length > 0) sequence.push(' ')
      sequence.push(element)
    }
    return uniqueOf(tokenViewOf(convSequence(sequence)))
  }

  it('drops a mixed token both sides hold', () => {
    const left = setOf([shared, onlyLeft])
    const right = setOf([shared, onlyRight])

    expect(difference(left, right)).toEqual([[onlyLeft]])
    expect(difference(right, left)).toEqual([[onlyRight]])
    expect(intersects(left, right)).toBe(true)
  })

  it('answers `has` for a packed key as well as a mixed one', () => {
    const packed = setOf([97, 98])
    const mixed = setOf([shared])
    const packedKey = [...packed.packed.keys()][0]
    const mixedKey = [...mixed.mixed.keys()][0]

    expect(packed.has(packedKey, [97])).toBe(true)
    expect(packed.has('no such key', [122])).toBe(false)
    expect(mixed.has(mixedKey, [shared])).toBe(true)
    expect(mixed.has(mixedKey, [onlyLeft])).toBe(false)
  })
})
