import { describe, expect, test } from 'vitest'
import {
  bestMatch,
  createMatcher,
  createScorer,
  isMatch,
  normalizeText,
  scoreIfMatch,
  scoreMatrix,
  scorePairs,
  search,
} from '../src/index.js'
import * as fuzz from '../src/fuzz.js'
import * as levenshtein from '../src/levenshtein.js'
import * as indel from '../src/indel.js'
import * as lcs from '../src/lcs.js'
import * as osa from '../src/osa.js'
import * as damerau from '../src/damerau-levenshtein.js'
import * as hamming from '../src/hamming.js'
import * as jaro from '../src/jaro.js'
import * as jaroWinkler from '../src/jaro-winkler.js'
import * as prefix from '../src/prefix.js'
import * as postfix from '../src/postfix.js'
import { withPublicScoreObserver } from '../src/scorer.js'

describe('0.6 metrics and scorers', () => {
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

    const invalid = createScorer(() => 6, {
      direction: 'similarity',
      bounds: [0, 5],
      symmetric: true,
    })
    expect(() => invalid.score('a', 'b', { threshold: 9 })).toThrow(RangeError)
  })

  test('scoreIfMatch and isMatch use scorer thresholds', () => {
    const scorer = createScorer(levenshtein.similarity)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBeCloseTo(2 / 3)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBeUndefined()
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBe(true)
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBe(false)
  })
})

describe('0.6 search and matrices', () => {
  const scorer = createScorer(fuzz.similarity)

  test('Matcher snapshots non-string sequences and retains original items and keys', () => {
    const text = ['a', 'b', 'c']
    const item = { text }
    const items = [item, null, { text: ['a', 'x', 'c'] }]
    const matcher = createMatcher(items, { scorer, getText: (value) => value?.text })
    text[0] = 'z'
    expect(matcher.size).toBe(2)
    expect(matcher.best(['a', 'b', 'c'])).toEqual({ item, key: 0, score: 100 })
    expect(matcher.search(['a', 'x', 'c'], { limit: null })[0]?.key).toBe(2)
  })

  test('maps and objects preserve keys while skipped values leave gaps', () => {
    const map = new Map([
      ['first', 'alpha'],
      ['missing', null],
      ['third', 'alpine'],
    ])
    const mapped = createMatcher(map, { scorer })
    expect(mapped.best('alpha')?.key).toBe('first')
    const object = createMatcher({ a: 'alpha', b: null, c: 'alpine' }, { scorer })
    expect(object.search('alpine', { limit: null }).map((match) => match.key)).toEqual([
      'c',
      'a',
    ])
  })

  test('one-shot and Matcher results agree and normalize once per retained value', () => {
    const items = ['Alpha', null, 'Alpine', 'Beta']
    const matcher = createMatcher(items, { scorer, normalize: normalizeText })
    expect(bestMatch('alp', items, { scorer, normalize: normalizeText })).toEqual(
      matcher.best('alp'),
    )
    expect(search('alp', items, { scorer, normalize: normalizeText, limit: null })).toEqual(
      matcher.search('alp', { limit: null }),
    )
  })

  test('Matcher candidate drivers never access the public score method', () => {
    let accesses = 0
    const observed = withPublicScoreObserver(scorer, () => {
      accesses++
    })
    const matcher = createMatcher(['alpha', 'beta'], { scorer: observed })
    expect(matcher.best('alpha')?.score).toBe(100)
    expect(matcher.search('alpha')).toHaveLength(2)
    expect(accesses).toBe(0)
  })

  test('matrix operations consume Scorer objects', () => {
    const normalized = createScorer(levenshtein.similarity)
    expect(scoreMatrix(['a', 'b'], ['a', 'c'], { scorer: normalized }).toArray()).toEqual([
      [1, 0],
      [0, 0],
    ])
    expect([...scorePairs(['a', 'b'], ['a', 'c'], { scorer: normalized })]).toEqual([
      1, 0,
    ])
  })

  test('collection policies and call limits are validated', () => {
    expect(() => createMatcher([null], { scorer, missingItems: 'throw' })).toThrow(
      TypeError,
    )
    expect(() =>
      createMatcher(['a'], { scorer, normalize: () => null }),
    ).toThrow(TypeError)
    const matcher = createMatcher(['a'], { scorer })
    expect(() => matcher.search('a', { limit: -1 })).toThrow(RangeError)
    expect(() => matcher.best('a', { threshold: Infinity })).toThrow(RangeError)
  })
})
