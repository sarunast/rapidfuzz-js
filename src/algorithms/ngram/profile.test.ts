// Ported from RapidFuzz's own suite; see tests/algorithms/dice.test.ts and
// cosine.test.ts for the metric-level assertions these support.
import { describe, expect, it } from 'vitest'

import { Cosine, Dice } from '../../../testing/scorers.js'
import { elementsEqual } from '../../core/sequence.js'
import { cosineSimilarity } from '../cosine/implementation.js'
import { diceSimilarity } from '../dice/implementation.js'
import { dotProduct, sharedFrequency } from './compare.js'
import { buildProfile } from './profile.js'

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

  it('specializes gram sizes 1 to 3 without changing the answer', () => {
    // gramSize 1, 2 and 3 take literal loops; 4 and up take the explicit stack.
    // The four paths have to agree wherever they overlap.
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

  it('leaves a trigram unmatched at whichever level first diverges', () => {
    // The literal trigram loop misses at two levels, and each has its own exit:
    // an absent first element never reaches the second, and an absent second
    // never reaches the counts.
    const abc = buildProfile('abc', 3)
    for (const other of ['xbc', 'axc']) {
      const diverging = buildProfile(other, 3)
      expect(sharedFrequency(abc, diverging), other).toBe(0)
      expect(dotProduct(abc, diverging), other).toBe(0)
    }
    // Both levels match and only the last element differs, which is where the
    // counts of two reachable leaves are compared.
    expect(sharedFrequency(abc, buildProfile('abx', 3))).toBe(0)
    expect(dotProduct(buildProfile('abcabc', 3), buildProfile('abcab', 3))).toBe(4)
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

  it('reaches back a whole window for a leading NaN at depth 3', () => {
    // The depth-3 builder unrolls the scan of the elements before the first
    // window, so which of the two it finds — and that a later one wins — is
    // worth pinning: an off-by-one here is a wrong `squaredNorm`, not a crash.
    const second = buildProfile(['a', Number.NaN, 'c', 'd', 'e'], 3)
    expect(second.gramCount).toBe(3)
    // Two windows reach the NaN and count only toward the norm; `cde` is the
    // one gram inserted.
    expect(second.squaredNorm).toBe(3)
    expect(sharedFrequency(second, buildProfile(['x', 'y', 'c', 'd', 'e'], 3))).toBe(1)

    const first = buildProfile([Number.NaN, 'b', 'c', 'd'], 3)
    expect(first.gramCount).toBe(2)
    expect(first.squaredNorm).toBe(2)
    expect(sharedFrequency(first, buildProfile(['b', 'c', 'd'], 3))).toBe(1)
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
