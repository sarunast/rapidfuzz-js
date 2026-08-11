import { describe, expect, it } from 'vitest'

import {
  distance as damerauDistance,
  similarity as damerauSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import {
  distance as hammingDistance,
  similarity as hammingSimilarity,
} from '../../src/algorithms/hamming/index.js'
import {
  distance as indelDistance,
  similarity as indelSimilarity,
} from '../../src/algorithms/indel/index.js'
import { similarity as jaroSimilarity } from '../../src/algorithms/jaro/index.js'
import { similarity as jaroWinklerSimilarity } from '../../src/algorithms/jaroWinkler/index.js'
import {
  distance as lcsDistance,
  similarity as lcsSimilarity,
} from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  similarity as levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/index.js'
import {
  distance as osaDistance,
  similarity as osaSimilarity,
} from '../../src/algorithms/osa/index.js'
import {
  distance as postfixDistance,
  similarity as postfixSimilarity,
} from '../../src/algorithms/postfix/index.js'
import {
  distance as prefixDistance,
  similarity as prefixSimilarity,
} from '../../src/algorithms/prefix/index.js'
import { createScorer } from '../../src/core/scorer.js'
import type { MaybeSequence } from '../../src/core/types.js'
import { callUntyped } from '../support/common.js'

type SimilarityMetric = (a: MaybeSequence, b: MaybeSequence) => number
type DistanceMetric = (a: MaybeSequence, b: MaybeSequence) => number

const SIMILARITIES: ReadonlyArray<readonly [string, SimilarityMetric]> = [
  ['Damerau-Levenshtein', damerauSimilarity],
  ['Hamming', hammingSimilarity],
  ['Indel', indelSimilarity],
  ['Jaro', jaroSimilarity],
  ['Jaro-Winkler', jaroWinklerSimilarity],
  ['LCS', lcsSimilarity],
  ['Levenshtein', levenshteinSimilarity],
  ['OSA', osaSimilarity],
  ['Postfix', postfixSimilarity],
  ['Prefix', prefixSimilarity],
]

const DISTANCES: ReadonlyArray<readonly [string, DistanceMetric]> = [
  ['Damerau-Levenshtein', damerauDistance],
  ['Hamming', hammingDistance],
  ['Indel', indelDistance],
  ['LCS', lcsDistance],
  ['Levenshtein', levenshteinDistance],
  ['OSA', osaDistance],
  ['Postfix', postfixDistance],
  ['Prefix', prefixDistance],
]

describe('canonical algorithm scales', () => {
  it('keeps normalized similarities in zero to one', () => {
    for (const [name, similarity] of SIMILARITIES) {
      expect(similarity('', ''), `${name}: empty identity`).toBe(1)
      const score = similarity('South Korea', 'North Korea')
      expect(score, name).toBeGreaterThanOrEqual(0)
      expect(score, name).toBeLessThanOrEqual(1)
      expect(similarity('same', 'same'), `${name}: identity`).toBe(1)
    }
  })

  it('keeps distances in native units', () => {
    expect(levenshteinDistance('kitten', 'sitting')).toBe(3)
    expect(indelDistance('South Korea', 'North Korea')).toBe(4)
    expect(lcsDistance('qabxcd', 'abycdf')).toBe(2)
    expect(osaDistance('CA', 'AC')).toBe(1)
    expect(damerauDistance('CA', 'ABC')).toBe(2)
    expect(hammingDistance('abc', 'axcd')).toBe(2)
    expect(prefixDistance('abcd', 'abcee')).toBe(2)
    expect(postfixDistance('abcd', 'eebcd')).toBe(2)
  })

  it('uses normalized similarity rather than maximum-minus-distance', () => {
    expect(levenshteinSimilarity('abc', 'axc')).toBeCloseTo(2 / 3, 12)
    expect(lcsSimilarity('abc', 'axc')).toBeCloseTo(2 / 3, 12)
    expect(indelSimilarity('abc', 'axc')).toBeCloseTo(2 / 3, 12)
    expect(prefixSimilarity('abcd', 'abxy')).toBe(0.5)
    expect(postfixSimilarity('xycd', 'abcd')).toBe(0.5)
  })
})

describe('canonical sequence handling', () => {
  it('treats compatible missing similarities as zero', () => {
    for (const [name, similarity] of SIMILARITIES) {
      expect(similarity(null, 'test'), name).toBe(0)
      expect(similarity('test', undefined), name).toBe(0)
    }
  })

  it('rejects missing distance inputs', () => {
    for (const [name, distance] of DISTANCES) {
      expect(() => distance(null, 'test'), name).toThrow(TypeError)
      expect(() => distance('test', undefined), name).toThrow(TypeError)
    }
  })

  it('treats array-like inputs consistently with strings', () => {
    const text = 'the wonderful new york mets'
    const characters = Array.from(text)
    const bytes = new TextEncoder().encode(text)
    for (const [name, similarity] of SIMILARITIES) {
      expect(similarity(characters, characters), `${name}: arrays`).toBe(1)
      expect(similarity(text, characters), `${name}: mixed`).toBe(1)
      expect(similarity(bytes, bytes), `${name}: typed arrays`).toBe(1)
    }
  })

  it('rejects values that are not sequences', () => {
    for (const invalid of [123, Number.NaN, true, Symbol('s'), { a: 1 }]) {
      expect(() => callUntyped(levenshteinSimilarity, invalid, 'test')).toThrow(TypeError)
    }
  })
})

describe('canonical scorer thresholds', () => {
  it('uses native distance thresholds', () => {
    const scorer = createScorer(levenshteinDistance)
    expect(scorer.score('kitten', 'sitting', { threshold: 3 })).toBe(3)
    expect(scorer.score('kitten', 'sitting', { threshold: 2 })).toBeUndefined()
  })

  it('uses normalized similarity thresholds', () => {
    const scorer = createScorer(levenshteinSimilarity)
    const exact = scorer.score('abc', 'axc')
    expect(scorer.score('abc', 'axc', { threshold: 2 / 3 })).toBe(exact)
    expect(scorer.score('abc', 'axc', { threshold: 0.7 })).toBeUndefined()
  })

  it('validates thresholds once at the scorer boundary', () => {
    const scorer = createScorer(jaroSimilarity)
    for (const threshold of [Number.NaN, Infinity, -Infinity]) {
      expect(() => scorer.score('abc', 'abd', { threshold })).toThrow(RangeError)
    }
  })
})
