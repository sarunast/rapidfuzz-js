import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { referenceDot, referenceShared } from '../../../../testing/reference/ngram.js'
import { Dice } from '../../../../testing/scorers.js'
import { directSharedFrequency, dotProduct, sharedFrequency } from './compare.js'
import { buildProfile, profileOfElements } from './profile.js'

describe('the transient direct counter', () => {
  // The counter returns a *count*, so it is checked against the profiles as an
  // integer rather than through a score within a tolerance: a rounding-sized
  // disagreement would be a wrong multiset, and `toBe` is what says so.
  // Long enough that every depth up to six has grams to intersect: a pair the
  // counter declines for having none tests the decline, not the count, and the
  // declines have their own cases below.
  const bounds = { minLength: 7, maxLength: 24 }
  const packableNumbers = fc.array(fc.integer({ min: 0, max: 0xff }), bounds)
  const packableChars = fc.array(fc.constantFrom('a', 'b', 'c', 'd'), bounds)
  // Each canonical radix and the first value above it, so every depth draws
  // elements from both sides of its own boundary: `0xff`/`0x100` at depths 4-6,
  // `0xffff`/`0x10000` at 3, `0x10ffff`/`0x110000` at 1-2.
  const radixEdges = fc.array(
    fc.constantFrom(0, 1, 0xff, 0x100, 0xffff, 0x1_0000, 0x10_ffff, 0x11_0000),
    bounds,
  )
  // Both sides drawn from one domain. Drawing them independently spent a large
  // share of the runs on cross-domain pairs, which decline before counting
  // anything — a decline this file tests directly, and which tells the count
  // nothing.
  const sameDomainPair = fc.oneof(
    fc.tuple(packableNumbers, packableNumbers),
    fc.tuple(packableChars, packableChars),
    fc.tuple(radixEdges, radixEdges),
  )

  it('counts exactly what the two profiles intersect to', () => {
    fc.assert(
      fc.property(
        sameDomainPair,
        fc.integer({ min: 1, max: 6 }),
        ([left, right], gramSize) => {
          const counted = directSharedFrequency(left, right, gramSize)
          if (counted === null) return
          expect(counted).toBe(
            sharedFrequency(
              profileOfElements(left, gramSize),
              profileOfElements(right, gramSize),
            ),
          )
          // Symmetric, and the tally side is chosen by length rather than by
          // argument order — so both orientations must reach one number.
          expect(directSharedFrequency(right, left, gramSize)).toBe(counted)
        },
      ),
      { numRuns: 4000 },
    )
  })

  it('declines a pair of mixed domains that still shares a gram, rather than answering zero', () => {
    // The domains come from the *first* element of each side, so a pair can be
    // refused here and still have grams in common: both of these hold `[1, 2]`.
    // Turning this decline into a `0` — which a fully homogeneous mixed pair
    // would deserve — silently loses that gram.
    const withChar = ['x', 1, 2]
    const withNumber = [9, 1, 2]
    expect(directSharedFrequency(withChar, withNumber, 2)).toBeNull()
    expect(directSharedFrequency(withNumber, withChar, 2)).toBeNull()
    expect(
      sharedFrequency(profileOfElements(withChar, 2), profileOfElements(withNumber, 2)),
    ).toBe(1)
    expect(Dice.similarity(withChar, withNumber, { gramSize: 2 })).toBeCloseTo(0.5, 12)
  })

  it('counts a repeated gram only as often as the smaller side holds it', () => {
    // `min(3, 7)`: the larger side keeps meeting a gram whose count has already
    // run out, which is the decrement's whole purpose.
    expect(directSharedFrequency('aaaa', 'aaaaaaaa', 2)).toBe(3)
    expect(directSharedFrequency('aaaaaaaa', 'aaaa', 2)).toBe(3)
    // And a gram the smaller side never held at all.
    expect(directSharedFrequency('abab', 'cdcdcdcd', 2)).toBe(0)
  })

  it('stops once the smaller side is spent, without loosening what it refuses', () => {
    // `ab`, `bc` and `cd` are all found in the larger side's first three grams,
    // and nothing in the six after them can raise a count that has already
    // reached the smaller side's total.
    expect(directSharedFrequency('abcd', 'abcdefghij', 2)).toBe(3)
    // The saturating walk must not turn into a licence to skip validation: the
    // keys are packed before any of them is counted, so an element the packing
    // refuses still declines the whole pair even when it sits past the point
    // the count stopped at. `null`, not `3`.
    const refusingTail = [...'abcdefghij', 'zz']
    expect(directSharedFrequency([...'abcd'], refusingTail, 2)).toBeNull()
    expect(directSharedFrequency(refusingTail, [...'abcd'], 2)).toBeNull()
  })

  it('spends every occurrence before it stops, not every distinct gram', () => {
    // `aaaaabbb` is `aa` four times, `ab` once and `bb` twice — seven gram
    // occurrences over three distinct grams. The larger side supplies them one
    // at a time, separated by `c`, and its very last gram is the second `bb`.
    //
    // So the stop condition has to be counting occurrences: a walk that stopped
    // once it had seen all three distinct grams would answer 3, and one that
    // stopped a gram early would answer 6.
    const small = 'aaaaabbb'
    expect(directSharedFrequency(small, 'aacaacaacaacabcbbcbb', 2)).toBe(7)
    // The same larger side one `bb` short, so the count never saturates and the
    // walk runs to the end: six of the seven.
    expect(directSharedFrequency(small, 'aacaacaacaacabcbb', 2)).toBe(6)
  })

  it('declines every shape it cannot pack, leaving the profiles to answer', () => {
    // No rung reaches seven elements.
    expect(
      directSharedFrequency([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], 7),
    ).toBeNull()
    // Nothing to count on one side or the other.
    expect(directSharedFrequency([1], [1, 2, 3], 2)).toBeNull()
    expect(directSharedFrequency([1, 2, 3], [1], 2)).toBeNull()
    // Two domains: `'b'` and `98` would become one key.
    expect(directSharedFrequency(['a', 'b', 'c'], [97, 98, 99], 2)).toBeNull()
    expect(directSharedFrequency([97, 98, 99], ['a', 'b', 'c'], 2)).toBeNull()
    // An unpackable element, on the side that is tallied and on the side that is
    // spent — the two sides are packed by separate calls.
    const long = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    const short = ['a', '😀', 'c']
    expect(directSharedFrequency(short, long, 3)).toBeNull()
    expect(directSharedFrequency(long, short, 3)).toBeNull()
    expect(directSharedFrequency(long.concat(short), long, 3)).toBeNull()
  })

  it('is invisible to the public score, on either side of the threshold', () => {
    // 512 grams is where Dice routes bigrams and trigrams to the counter, and
    // every other depth stays on the profiles however long the input is. The
    // score may not notice either way, so all six are checked at a length past
    // the threshold — which depth is routed is a benchmark's business.
    for (const gramSize of [1, 2, 3, 4, 5, 6]) {
      for (const length of [511, 512, 513, 700]) {
        const text = 'abcdefghij'.repeat(Math.ceil(length / 10)).slice(0, length)
        const edited = `${text.slice(0, 100)}z${text.slice(101)}`
        expect(
          Dice.similarity(text, edited, { gramSize }),
          `gramSize ${gramSize}, ${length} chars`,
        ).toBe(
          (2 *
            sharedFrequency(
              profileOfElements(Array.from(text), gramSize),
              profileOfElements(Array.from(edited), gramSize),
            )) /
            (text.length - gramSize + 1 + (edited.length - gramSize + 1)),
        )
      }
    }
    // Above the threshold and unpackable: the counter declines and the profiles
    // answer, which the score cannot tell from never having tried.
    const astral = `${'ab'.repeat(400)}😀`
    expect(Dice.similarity(astral, astral, { gramSize: 3 })).toBe(1)
    // 801 code points, so 799 trigrams, and only the last of them reaches the
    // emoji: the other 798 all match, against 798 on the plain side.
    expect(Dice.similarity(astral, 'ab'.repeat(400), { gramSize: 3 })).toBeCloseTo(
      (2 * 798) / (799 + 798),
      12,
    )
  })
})

describe('properties', () => {
  const sequences = fc.string({ maxLength: 24 })
  const gramSizes = fc.integer({ min: 1, max: 5 })
  // Every element kind the trie has to keep apart, including the two the `Map`
  // and `===` disagree about.
  const shared = { unused: 0 }
  const elements = fc.oneof(
    fc.constantFrom('a', 'b', 'ab', '', 'a,b'),
    fc.constantFrom(0, -0, 1, 2, Number.NaN),
    fc.constantFrom(true, false, null, undefined),
    fc.constantFrom(shared, { unused: 0 }),
  )

  it('agrees with a reference n-gram counter over arbitrary elements', () => {
    fc.assert(
      fc.property(
        fc.array(elements, { maxLength: 14 }),
        fc.array(elements, { maxLength: 14 }),
        gramSizes,
        (left, right, gramSize) => {
          const a = buildProfile(left, gramSize)
          const b = buildProfile(right, gramSize)
          expect(sharedFrequency(a, b)).toBe(referenceShared(left, right, gramSize))
          expect(dotProduct(a, b)).toBe(referenceDot(left, right, gramSize))
        },
      ),
      { numRuns: 500 },
    )
  })

  it('intersects symmetrically and within the smaller profile', () => {
    fc.assert(
      fc.property(sequences, sequences, gramSizes, (left, right, gramSize) => {
        const a = buildProfile(left, gramSize)
        const b = buildProfile(right, gramSize)
        expect(sharedFrequency(a, b)).toBe(sharedFrequency(b, a))
        expect(dotProduct(a, b)).toBe(dotProduct(b, a))
        expect(sharedFrequency(a, b)).toBeLessThanOrEqual(
          Math.min(a.gramCount, b.gramCount),
        )
        expect(dotProduct(a, b)).toBeGreaterThanOrEqual(0)
        expect(sharedFrequency(a, a)).toBe(a.gramCount)
        expect(dotProduct(a, a)).toBe(a.squaredNorm)
      }),
    )
  })
})
