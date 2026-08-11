import { describe, expect, test } from 'vitest'

import * as damerau from '../../src/algorithms/damerauLevenshtein/index.js'
import * as hamming from '../../src/algorithms/hamming/index.js'
import * as indel from '../../src/algorithms/indel/index.js'
import * as jaro from '../../src/algorithms/jaro/index.js'
import * as jaroWinkler from '../../src/algorithms/jaroWinkler/index.js'
import * as lcs from '../../src/algorithms/lcs/index.js'
import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import * as osa from '../../src/algorithms/osa/index.js'
import * as postfix from '../../src/algorithms/postfix/index.js'
import * as prefix from '../../src/algorithms/prefix/index.js'
import { scorerCompilation } from '../../src/core/scorer.js'
import * as fuzz from '../../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  isMatch,
  scoreIfMatch,
} from '../../src/index.js'

describe('Metric and Scorer contracts', () => {
  test('algorithm families keep their natural scales', () => {
    expect(fuzz.similarity('abc', 'axc')).toBeCloseTo(200 / 3)
    for (const metric of [
      levenshtein.similarity,
      indel.similarity,
      lcs.similarity,
      osa.similarity,
      damerau.similarity,
      hamming.similarity,
      prefix.similarity,
      postfix.similarity,
      jaro.similarity,
      jaroWinkler.similarity,
    ]) {
      expect(metric('same', 'same')).toBe(1)
      expect(metric('a', 'b')).toBeGreaterThanOrEqual(0)
      expect(metric('a', 'b')).toBeLessThanOrEqual(1)
    }
    expect(levenshtein.distance('abc', 'axc')).toBe(1)
    expect(indel.distance('abc', 'axc')).toBe(2)
    expect(lcs.distance('abc', 'axc')).toBe(1)
    expect(osa.distance('ab', 'ba')).toBe(1)
    expect(damerau.distance('ab', 'ba')).toBe(1)
    expect(hamming.distance('abc', 'axc')).toBe(1)
    for (const metric of [
      fuzz.partialSimilarity,
      fuzz.tokenSortSimilarity,
      fuzz.tokenSetSimilarity,
      fuzz.tokenSimilarity,
      fuzz.partialTokenSortSimilarity,
      fuzz.partialTokenSetSimilarity,
      fuzz.partialTokenSimilarity,
      fuzz.fuzzySimilarity,
    ]) {
      expect(metric('new york mets', 'new york mets')).toBeGreaterThanOrEqual(0)
    }
    expect(fuzz.partialSimilarityAlignment('abc', 'zabc')?.score).toBe(100)
  })

  test('scorer metadata, configuration, freezing, and thresholds are scale aware', () => {
    const fuzzy = createScorer(fuzz.fuzzySimilarity)
    const normalized = createScorer(levenshtein.similarity, {
      weights: { insertion: 1, deletion: 2, substitution: 1 },
    })
    const distance = createScorer(levenshtein.distance, {
      weights: [1, 2, 1],
    })
    expect(Object.isFrozen(fuzzy)).toBe(true)
    expect(fuzzy.bounds).toEqual([0, 100])
    expect(normalized.bounds).toEqual([0, 1])
    expect(normalized.symmetric).toBe(false)
    expect(distance.bounds).toEqual([0, Number.POSITIVE_INFINITY])
    expect(fuzzy.score('abc', 'axc', { threshold: 60 })).toBeCloseTo(200 / 3)
    expect(fuzzy.score('abc', 'axc', { threshold: 80 })).toBeUndefined()
    expect(normalized.score('abc', 'axc', { threshold: 0.6 })).toBeCloseTo(2 / 3)
    expect(normalized.score('abc', 'axc', { threshold: 0.8 })).toBeUndefined()
    expect(distance.score('abc', 'axc', { threshold: 1 })).toBe(1)
    expect(distance.score('abc', 'axc', { threshold: 0 })).toBeUndefined()
    for (const threshold of [Number.NaN, Infinity, -Infinity]) {
      expect(() => fuzzy.score('a', 'a', { threshold })).toThrow(RangeError)
    }
    expect(fuzzy.score('a', 'a', { threshold: 101 })).toBeUndefined()
  })

  test('missing and invalid inputs follow the scorer direction', () => {
    const compatible = createScorer(levenshtein.similarity)
    const throwing = createScorer(levenshtein.similarity, { missing: 'throw' })
    const distance = createScorer(levenshtein.distance)
    expect(compatible.score(null, 'abc')).toBe(0)
    expect(compatible.score(undefined, 'abc')).toBe(0)
    expect(() => throwing.score(null, 'abc')).toThrow(TypeError)
    expect(() => distance.score(null, 'abc')).toThrow(TypeError)
    for (const invalid of [1, true, Number.NaN, {}]) {
      expect(() => Reflect.apply(compatible.score, compatible, [invalid, 'abc'])).toThrow(
        TypeError,
      )
    }
    expect(compatible.score('', '')).toBe(1)
    expect(levenshtein.similarity(null, 'abc')).toBe(0)
  })

  test('custom results are always validated and custom bounds never prune execution', () => {
    let calls = 0
    const custom = createScorer(
      (a, b) => {
        calls++
        return a === b ? 5 : 2
      },
      { direction: 'similarity', bounds: [0, 5], symmetric: true },
    )
    expect(custom.score('a', 'b', { threshold: 9 })).toBeUndefined()
    expect(calls).toBe(1)
    expect(custom.score(null, 'b')).toBe(0)

    const invalid = createScorer(() => 6, {
      direction: 'similarity',
      bounds: [0, 5],
      symmetric: true,
    })
    expect(() => invalid.score('a', 'b', { threshold: 9 })).toThrow(RangeError)

    calls = 0
    expect(
      bestMatch('a', ['a', 'b'], {
        scorer: custom,
      }),
    ).toEqual({ item: 'a', key: 0, score: 5 })
    expect(calls).toBe(2)
    expect(createMatcher(['a', 'b'], { scorer: custom }).best('a')).toEqual({
      item: 'a',
      key: 0,
      score: 5,
    })

    for (const result of [Number.NaN, Infinity, -1]) {
      const broken = createScorer(() => result, {
        direction: 'similarity',
        bounds: [0, 1],
        symmetric: true,
      })
      expect(() => broken.score('a', 'b')).toThrow(RangeError)
    }
    expect(() =>
      Reflect.apply(createScorer, undefined, [() => 1, { direction: 'similarity' }]),
    ).toThrow(TypeError)
    expect(() => Reflect.apply(createScorer, undefined, [() => 1])).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'distance',
          bounds: 'no bounds',
          symmetric: true,
        },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'similarity',
          bounds: [0, 1],
          symmetric: 'yes',
        },
      ]),
    ).toThrow(TypeError)
    for (const bounds of [
      ['zero', 1],
      [0, 'one'],
      [Infinity, Infinity],
      [0, Number.NaN],
      [2, 1],
    ]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [
          () => 1,
          {
            direction: 'similarity',
            bounds,
            symmetric: true,
          },
        ]),
      ).toThrow(RangeError)
    }
  })

  test('invalid built-in configuration is rejected at compilation', () => {
    expect(
      createScorer(levenshtein.similarity, { missing: 'compatible' }).score(null, 'a'),
    ).toBe(0)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        levenshtein.similarity,
        { missing: 'nope' },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [levenshtein.distance, { weights: [1, 2] }]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        levenshtein.distance,
        {
          weights: [1, 'two', 3],
        },
      ]),
    ).toThrow(TypeError)
  })

  test('scoreIfMatch and isMatch use scorer thresholds', () => {
    const scorer = createScorer(levenshtein.similarity)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBeCloseTo(2 / 3)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBeUndefined()
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBe(true)
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBe(false)
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0 })).toBe(true)
    const custom = createScorer(() => 1, {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(isMatch(custom, 'a', 'b', { threshold: 0 })).toBe(true)
    const distance = createScorer(levenshtein.distance)
    expect(distance.score('a', 'a', { threshold: -1 })).toBeUndefined()
    expect(() => Reflect.apply(scorerCompilation, undefined, [{}])).toThrow(TypeError)
  })
})
