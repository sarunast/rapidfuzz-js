import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { cosineSimilarity } from '../../src/algorithms/cosine/implementation.js'
import { diceSimilarity } from '../../src/algorithms/dice/implementation.js'
import {
  buildProfile,
  dotProduct,
  elementsEqual,
  sharedFrequency,
} from '../../src/algorithms/shared/ngram.js'
import { Cosine, Dice } from '../support/scorers.js'

describe('the profile at every depth', () => {
  it('counts, norms and intersects a repeated gram', () => {
    // `aaaa` is one distinct bigram with a count of 3.
    const four = buildProfile('aaaa', 2)
    const three = buildProfile('aaa', 2)
    expect(four.gramCount).toBe(3)
    expect(four.squaredNorm).toBe(9)
    expect(three.gramCount).toBe(2)
    expect(three.squaredNorm).toBe(4)
    expect(sharedFrequency(four, three)).toBe(2)
    expect(dotProduct(four, three)).toBe(6)
    // Dice reads the counts, cosine reads the direction — and these two
    // frequency vectors are parallel.
    expect(diceSimilarity('aaaa', 'aaa')).toBeCloseTo(0.8, 12)
    expect(cosineSimilarity('aaaa', 'aaa')).toBe(1)
  })

  it('specializes unigrams and bigrams without changing the answer', () => {
    // gramSize 1 and 2 take literal loops; 3 and up take the explicit stack.
    // The three paths have to agree wherever they overlap.
    for (const gramSize of [1, 2, 3, 4, 9]) {
      const a = buildProfile('abcabcab', gramSize)
      const b = buildProfile('abcabc', gramSize)
      expect(sharedFrequency(a, b), `gramSize ${gramSize}`).toBe(sharedFrequency(b, a))
      expect(dotProduct(a, b), `gramSize ${gramSize}`).toBe(dotProduct(b, a))
    }
    expect(sharedFrequency(buildProfile('aba', 1), buildProfile('ab', 1))).toBe(2)
    expect(sharedFrequency(buildProfile('aba', 2), buildProfile('ab', 2))).toBe(1)
    expect(sharedFrequency(buildProfile('abcd', 3), buildProfile('abc', 3))).toBe(1)
  })

  it('has nothing to intersect when a profile has no grams', () => {
    for (const gramSize of [1, 2, 3, 5]) {
      const empty = buildProfile('', gramSize)
      const full = buildProfile('abcdef', gramSize)
      expect(empty.gramCount, `gramSize ${gramSize}`).toBe(0)
      expect(empty.squaredNorm, `gramSize ${gramSize}`).toBe(0)
      expect(sharedFrequency(empty, full), `gramSize ${gramSize}`).toBe(0)
      expect(sharedFrequency(full, empty), `gramSize ${gramSize}`).toBe(0)
      expect(dotProduct(empty, full), `gramSize ${gramSize}`).toBe(0)
      expect(dotProduct(empty, empty), `gramSize ${gramSize}`).toBe(0)
    }
    expect(sharedFrequency(buildProfile('ab', 3), buildProfile('abcd', 3))).toBe(0)
  })

  it('retains the elements only when there is nothing else to compare', () => {
    expect(buildProfile('ab', 3).elements).not.toBeNull()
    expect(buildProfile('abcd', 2).elements).toBeNull()
  })
})

describe('element equality', () => {
  it('treats a gram holding NaN as unmatchable at every window', () => {
    // Unmatchable grams still count toward `gramCount` and `squaredNorm`, so
    // the denominators stay right and cosine never divides by zero.
    const single = buildProfile([Number.NaN], 1)
    expect(single.gramCount).toBe(1)
    expect(single.squaredNorm).toBe(1)
    expect(sharedFrequency(single, single)).toBe(0)
    expect(dotProduct(single, single)).toBe(0)

    const pair = buildProfile([Number.NaN, Number.NaN], 1)
    expect(pair.gramCount).toBe(2)
    expect(pair.squaredNorm).toBe(2)
    expect(sharedFrequency(pair, pair)).toBe(0)

    // Both bigram windows of `[1, NaN, 1]` contain the NaN; at gramSize 1 only
    // the middle element does, and the two `1`s still match.
    const bigrams = buildProfile([1, Number.NaN, 1], 2)
    expect(bigrams.gramCount).toBe(2)
    expect(bigrams.squaredNorm).toBe(2)
    expect(sharedFrequency(bigrams, bigrams)).toBe(0)

    const unigrams = buildProfile([1, Number.NaN, 1], 1)
    expect(unigrams.gramCount).toBe(3)
    expect(unigrams.squaredNorm).toBe(5)
    expect(sharedFrequency(unigrams, unigrams)).toBe(2)
    expect(
      Dice.similarity([1, Number.NaN, 1], [1, Number.NaN, 1], { gramSize: 1 }),
    ).toBeCloseTo((2 * 2) / (3 + 3), 12)

    // A trailing NaN takes out the windows that reach it and no others.
    const trailing = buildProfile(['a', 'b', 'c', Number.NaN], 2)
    expect(trailing.gramCount).toBe(3)
    expect(sharedFrequency(trailing, buildProfile(['a', 'b', 'c'], 2))).toBe(2)
  })

  it('matches negative zero to positive zero', () => {
    // `===` and SameValueZero agree here, so no special handling is needed and
    // none is present — this pins that.
    expect(elementsEqual([-0], [0])).toBe(true)
    expect(sharedFrequency(buildProfile([-0], 1), buildProfile([0], 1))).toBe(1)
    expect(Dice.similarity([-0, 1], [0, 1], { gramSize: 1 })).toBe(1)
    expect(Cosine.similarity([-0, 1], [0, 1], { gramSize: 1 })).toBe(1)
  })

  it('compares objects by identity and strings by value', () => {
    const token = {}
    expect(sharedFrequency(buildProfile([token], 1), buildProfile([token], 1))).toBe(1)
    expect(sharedFrequency(buildProfile([token], 1), buildProfile([{}], 1))).toBe(0)
    expect(sharedFrequency(buildProfile(['ab'], 1), buildProfile(['ab'], 1))).toBe(1)
  })

  it('agrees between a string and its code points', () => {
    const text = 'the wonderful 😀 mets'
    const codePoints = Array.from(text)
    expect(diceSimilarity(text, codePoints)).toBe(1)
    expect(cosineSimilarity(text, codePoints)).toBe(1)
    expect(sharedFrequency(buildProfile(text, 2), buildProfile(codePoints, 2))).toBe(
      buildProfile(text, 2).gramCount,
    )
  })
})

/**
 * A profile the trie is checked against: one bucket per gram, keyed by a
 * position-aware structure rather than a serialized string, and using the
 * library's `===` so a gram holding `NaN` matches nothing — including itself.
 */
function referenceGrams(
  elements: readonly unknown[],
  gramSize: number,
): Array<{ readonly gram: readonly unknown[]; count: number }> {
  const buckets: Array<{ readonly gram: readonly unknown[]; count: number }> = []
  for (let start = 0; start + gramSize <= elements.length; start++) {
    const gram = elements.slice(start, start + gramSize)
    if (gram.some((element) => typeof element === 'number' && Number.isNaN(element))) {
      continue
    }
    const existing = buckets.find(
      (bucket) =>
        bucket.gram.length === gram.length &&
        bucket.gram.every((element, index) => element === gram[index]),
    )
    if (existing === undefined) buckets.push({ gram, count: 1 })
    else existing.count++
  }
  return buckets
}

function referenceShared(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let shared = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) shared += Math.min(bucket.count, match.count)
  }
  return shared
}

function referenceDot(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let product = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) product += bucket.count * match.count
  }
  return product
}

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
