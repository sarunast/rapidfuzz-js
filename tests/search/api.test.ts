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
import { withPublicScoreObserver } from '../../src/core/scorer.js'
import * as fuzz from '../../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  normalizeText,
  search,
} from '../../src/index.js'

describe('one-shot search and Matcher', () => {
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
    expect(
      search('alp', items, { scorer, normalize: normalizeText, limit: null }),
    ).toEqual(matcher.search('alp', { limit: null }))
    expect(bestMatch('none', items, { scorer, threshold: 100 })).toBeUndefined()
    expect(search('alp', items, { scorer, limit: 0 })).toEqual([])
    expect(search('same', ['same', 'same'], { scorer, limit: null })).toEqual([
      { item: 'same', key: 0, score: 100 },
      { item: 'same', key: 1, score: 100 },
    ])
    expect(search('none', items, { scorer, threshold: 100, limit: null })).toEqual([])
  })

  test('distance scorers use best-first ordering and maximum thresholds', () => {
    const distance = createScorer(levenshtein.distance)
    const items = ['sitting', 'kitten', 'kitchen']
    const matcher = createMatcher(items, { scorer: distance })
    expect(matcher.best('kitten')).toEqual({ item: 'kitten', key: 1, score: 0 })
    expect(matcher.search('kitten', { threshold: 2, limit: null })).toEqual([
      { item: 'kitten', key: 1, score: 0 },
      { item: 'kitchen', key: 2, score: 2 },
    ])
    expect(bestMatch('kitten', items, { scorer: distance })).toEqual(
      matcher.best('kitten'),
    )
    expect(search('kitten', items, { scorer: distance, limit: 2 })).toEqual([
      { item: 'kitten', key: 1, score: 0 },
      { item: 'kitchen', key: 2, score: 2 },
    ])
    expect(search('a', ['a', 'b'], { scorer: distance, threshold: 0 })).toEqual([
      { item: 'a', key: 0, score: 0 },
    ])
  })

  test('every fuzzy scorer supports prepared repeated search', () => {
    const metrics = [
      fuzz.similarity,
      fuzz.partialSimilarity,
      fuzz.tokenSortSimilarity,
      fuzz.tokenSetSimilarity,
      fuzz.tokenSimilarity,
      fuzz.partialTokenSortSimilarity,
      fuzz.partialTokenSetSimilarity,
      fuzz.partialTokenSimilarity,
      fuzz.fuzzySimilarity,
    ]
    for (const metric of metrics) {
      const prepared = createMatcher(
        ['new york mets', 'the wonderful new york mets', 'mets new york', ''],
        { scorer: createScorer(metric) },
      )
      expect(
        prepared.search('new york mets', { threshold: 0, limit: null }),
      ).toHaveLength(4)
    }
  })

  test('every algorithm family supports its prepared choice protocol', () => {
    for (const metric of [
      damerau.similarity,
      indel.similarity,
      jaro.similarity,
      jaroWinkler.similarity,
      lcs.similarity,
      osa.similarity,
      hamming.similarity,
      prefix.similarity,
      postfix.similarity,
    ]) {
      const matcher = createMatcher(['alphabet', 'alphanumeric', 'beta'], {
        scorer: createScorer(metric),
      })
      expect(matcher.search('alphabet', { limit: null })).toHaveLength(3)
    }
    expect(lcs.editops('same', 'same').operations).toEqual([])
  })

  test('missing queries and streamed collections retain new search semantics', () => {
    const matcher = createMatcher(['alpha', 'beta'], { scorer })
    expect(matcher.best(null)).toEqual({ item: 'alpha', key: 0, score: 0 })
    expect(matcher.search(undefined, { limit: null })).toEqual([
      { item: 'alpha', key: 0, score: 0 },
      { item: 'beta', key: 1, score: 0 },
    ])
    function* values(): Generator<string> {
      yield 'beta'
      yield 'alpha'
    }
    expect(
      search('alpha', values(), { scorer, limit: null }).map((match) => match.key),
    ).toEqual([1, 0])
    expect(bestMatch(null, [null, 'alpha', 'beta'], { scorer })).toEqual({
      item: 'alpha',
      key: 1,
      score: 0,
    })
    expect(bestMatch(null, ['alpha'], { scorer, threshold: 1 })).toBeUndefined()
    expect(search(null, [null, 'alpha', 'beta'], { scorer, limit: 1 })).toEqual([
      { item: 'alpha', key: 1, score: 0 },
    ])
    expect(search(null, ['alpha'], { scorer, threshold: 1, limit: null })).toEqual([])
    expect(matcher.best(null, { threshold: 1 })).toBeUndefined()
    expect(matcher.search(null, { limit: 1 })).toEqual([
      { item: 'alpha', key: 0, score: 0 },
    ])
    expect(matcher.search(null, { threshold: 1, limit: null })).toEqual([])
    expect(createMatcher([], { scorer }).best(null)).toBeUndefined()
    expect(matcher.search('alpha', { limit: 0 })).toEqual([])
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
    expect(observed.score('alpha', 'alpha')).toBe(100)
    expect(accesses).toBe(1)
  })

  test('collection policies and call limits are validated', () => {
    expect(() => createMatcher([null], { scorer, missingItems: 'throw' })).toThrow(
      TypeError,
    )
    expect(() => createMatcher(['a'], { scorer, normalize: () => null })).toThrow(
      TypeError,
    )
    expect(() =>
      createMatcher([{ text: null }], {
        scorer,
        getText: (item) => item.text,
        missingItems: 'throw',
      }),
    ).toThrow(TypeError)
    expect(
      createMatcher([{ text: null }], { scorer, getText: (item) => item.text }).size,
    ).toBe(0)
    expect(() => bestMatch('query', [], { scorer, normalize: () => null })).toThrow(
      TypeError,
    )
    const matcher = createMatcher(['a'], { scorer })
    expect(() => matcher.search('a', { limit: -1 })).toThrow(RangeError)
    expect(() => matcher.search('a', { limit: 0.5 })).toThrow(RangeError)
    expect(() => matcher.best('a', { threshold: Infinity })).toThrow(RangeError)
    expect(() => Reflect.apply(createMatcher, undefined, ['a', { scorer }])).toThrow(
      TypeError,
    )
    expect(() => Reflect.apply(createMatcher, undefined, [5, { scorer }])).toThrow(
      TypeError,
    )
    expect(() => Reflect.apply(createMatcher, undefined, [null, { scorer }])).toThrow(
      TypeError,
    )
    const distance = createScorer(levenshtein.distance)
    expect(() => createMatcher([], { scorer: distance }).best(null)).toThrow(TypeError)
    expect(() => bestMatch(null, [], { scorer: distance })).toThrow(TypeError)
  })
})
