import { describe, expect, expectTypeOf, test } from 'vitest'

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
  searchIter,
} from '../../src/index.js'
import type { Match, Sequence } from '../../src/index.js'
import type { ItemIterable } from '../../src/search/types.js'

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

  test('snapshots an array-like length exactly once after validation', () => {
    let lengthReads = 0
    const text = {
      0: 'a',
      1: 'b',
      get length() {
        lengthReads++
        return 2
      },
    }
    const matcher = createMatcher([text], { scorer })
    expect(matcher.best(['a', 'b'])?.score).toBe(100)
    expect(lengthReads).toBe(2)
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

  test('one-shot operations infer collection key types', () => {
    const arrayBest = bestMatch('a', ['a'], { scorer })
    const mapBest = bestMatch('a', new Map([[Symbol.for('a'), 'a']]), { scorer })
    const objectResults = search('a', { first: 'a' }, { scorer })
    const iterableResults = search('a', new Set(['a']), { scorer })
    const mapStream = searchIter('a', new Map([[Symbol.for('a'), 'a']]), { scorer })
    expectTypeOf(arrayBest?.key).toEqualTypeOf<number | undefined>()
    expectTypeOf(mapBest?.key).toEqualTypeOf<symbol | undefined>()
    expectTypeOf(objectResults).toEqualTypeOf<readonly Match<string, string>[]>()
    expectTypeOf(iterableResults).toEqualTypeOf<readonly Match<string, number>[]>()
    expectTypeOf(mapStream).toEqualTypeOf<IterableIterator<Match<string, symbol>>>()

    expectTypeOf(createMatcher(['a'], { scorer }).best('a')).toEqualTypeOf<
      Match<string, number> | undefined
    >()
    expectTypeOf(createMatcher(new Map([[1, 'a']]), { scorer }).best('a')).toEqualTypeOf<
      Match<string, number> | undefined
    >()
    expectTypeOf(createMatcher({ first: 'a' }, { scorer }).best('a')).toEqualTypeOf<
      Match<string, string> | undefined
    >()
    // A string is an `Iterable<string>`, and the runtime refuses one; without
    // the `& object` in `ItemIterable` the two disagree and this compiles.
    expectTypeOf<string>().not.toExtend<ItemIterable<string>>()
    expectTypeOf<readonly string[]>().toExtend<ItemIterable<string>>()
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
    expect(bestMatch('none', items, { scorer, threshold: 101 })).toBeUndefined()
    expect(search('alp', items, { scorer, limit: 0 })).toEqual([])
    expect(search('same', ['same', 'same'], { scorer, limit: null })).toEqual([
      { item: 'same', key: 0, score: 100 },
      { item: 'same', key: 1, score: 100 },
    ])
    expect(search('none', items, { scorer, threshold: 100, limit: null })).toEqual([])
    expect(search('none', items, { scorer, threshold: 101, limit: null })).toEqual([])
  })

  test('searchIter is lazy, source ordered, and stops reading with its caller', () => {
    let reads = 0
    function* choices(): Generator<string> {
      reads++
      yield 'alpha'
      reads++
      yield 'alpine'
      reads++
      yield 'beta'
    }
    const iterator = searchIter('alpha', choices(), { scorer, threshold: 40 })
    expect(reads).toBe(0)
    expect(iterator.next()).toEqual({
      done: false,
      value: { item: 'alpha', key: 0, score: 100 },
    })
    expect(reads).toBe(1)
    if (iterator.return === undefined) throw new TypeError('iterator is not closable')
    iterator.return()
    expect(reads).toBe(1)

    expect(
      Array.from(searchIter('alpha', ['alpine', 'alpha', 'beta'], { scorer })),
    ).toEqual([
      { item: 'alpine', key: 0, score: 54.54545454545454 },
      { item: 'alpha', key: 1, score: 100 },
      { item: 'beta', key: 2, score: 22.22222222222222 },
    ])

    const longChoices: Array<string | null> = [
      'alpha',
      'alpha',
      'alpha',
      'alpha',
      'alpha',
      'alpha',
      'alpha',
      'alpha',
      null,
      'beta',
      'alpha',
    ]
    expect(
      Array.from(searchIter('alpha', longChoices, { scorer, threshold: 50 })).map(
        (match) => match.key,
      ),
    ).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 10])
  })

  test('searchIter applies missing, distance, keyed, and impossible thresholds', () => {
    expect(Array.from(searchIter(null, [null, 'alpha'], { scorer }))).toEqual([
      { item: 'alpha', key: 1, score: 0 },
    ])
    expect(
      Array.from(
        searchIter(
          null,
          new Map<string, string | null>([
            ['missing', null],
            ['alpha', 'alpha'],
          ]),
          {
            scorer,
          },
        ),
      ),
    ).toEqual([{ item: 'alpha', key: 'alpha', score: 0 }])
    expect(Array.from(searchIter(null, ['alpha'], { scorer, threshold: 1 }))).toEqual([])
    expect(
      Array.from(searchIter('alpha', ['alpha'], { scorer, threshold: 101 })),
    ).toEqual([])
    expect(Array.from(searchIter('alpha', [null, 'alpha'], { scorer }))).toEqual([
      { item: 'alpha', key: 1, score: 100 },
    ])
    expect(
      Array.from(
        searchIter(
          'alpha',
          new Map<string, string | null>([
            ['missing', null],
            ['poor', 'beta'],
            ['exact', 'alpha'],
          ]),
          { scorer, threshold: 50 },
        ),
      ),
    ).toEqual([{ item: 'alpha', key: 'exact', score: 100 }])

    const distance = createScorer(levenshtein.distance)
    expect(
      Array.from(
        searchIter('kitten', ['sitting', 'kitten', 'kitchen'], {
          scorer: distance,
          threshold: 2,
        }),
      ),
    ).toEqual([
      { item: 'kitten', key: 1, score: 0 },
      { item: 'kitchen', key: 2, score: 2 },
    ])
    expect(() => searchIter(null, [], { scorer: distance }).next()).toThrow(TypeError)
    expect(() => searchIter('a', ['a'], { scorer, threshold: Infinity }).next()).toThrow(
      RangeError,
    )
  })

  test('Matcher searchIter reuses stored choices without public scoring', () => {
    let accesses = 0
    const observed = withPublicScoreObserver(scorer, () => {
      accesses++
    })
    const matcher = createMatcher(['alpine', 'alpha', 'beta'], { scorer: observed })
    expect(Array.from(matcher.searchIter('alpha', { threshold: 50 }))).toEqual([
      { item: 'alpine', key: 0, score: 54.54545454545454 },
      { item: 'alpha', key: 1, score: 100 },
    ])
    expect(Array.from(matcher.searchIter(null))).toHaveLength(3)
    expect(Array.from(matcher.searchIter(null, { threshold: 1 }))).toEqual([])
    expect(Array.from(matcher.searchIter('alpha', { threshold: 101 }))).toEqual([])
    const distance = createMatcher(['sitting', 'kitten'], {
      scorer: createScorer(levenshtein.distance),
    })
    expect(Array.from(distance.searchIter('kitten', { threshold: 1 }))).toEqual([
      { item: 'kitten', key: 1, score: 0 },
    ])
    const custom = createScorer((left, right) => (left === right ? 1 : 0), {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(
      Array.from(createMatcher(['a', 'b'], { scorer: custom }).searchIter('a')),
    ).toEqual([
      { item: 'a', key: 0, score: 1 },
      { item: 'b', key: 1, score: 0 },
    ])
    expect(Array.from(searchIter('a', ['a', 'b'], { scorer: custom }))).toEqual([
      { item: 'a', key: 0, score: 1 },
      { item: 'b', key: 1, score: 0 },
    ])
    expect(accesses).toBe(0)
  })

  test('Matcher snapshots construction options before repeated queries', () => {
    const items = [{ primary: 'Alpha', alternate: 'Wrong' }]
    const options: {
      scorer: typeof scorer
      getText: (item: (typeof items)[number]) => Sequence
      normalize: (value: Sequence) => Sequence
      missingItems: 'skip' | 'throw'
    } = {
      scorer,
      getText: (item) => item.primary,
      normalize: (value) => String(value).toLowerCase(),
      missingItems: 'skip',
    }
    const matcher = createMatcher(items, options)

    options.getText = (item) => item.alternate
    options.normalize = (value) => String(value).toUpperCase()
    options.missingItems = 'throw'

    expect(matcher.best('ALPHA')).toEqual({ item: items[0], key: 0, score: 100 })
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
    expect(bestMatch(null, [null], { scorer })).toBeUndefined()
    expect(bestMatch(null, new Map([['only', 'alpha']]), { scorer })?.key).toBe('only')
    expect(
      bestMatch(
        null,
        new Map<string, string | null>([
          ['missing', null],
          ['only', 'alpha'],
        ]),
        {
          scorer,
        },
      )?.key,
    ).toBe('only')
    expect(bestMatch(null, new Map(), { scorer })).toBeUndefined()
    expect(search(null, [null, 'alpha', 'beta'], { scorer, limit: 1 })).toEqual([
      { item: 'alpha', key: 1, score: 0 },
    ])
    expect(search(null, [null, 'alpha', 'beta'], { scorer, limit: 2 })).toEqual([
      { item: 'alpha', key: 1, score: 0 },
      { item: 'beta', key: 2, score: 0 },
    ])
    expect(search('none', ['alpha'], { scorer, threshold: 100, limit: 1 })).toEqual([])
    expect(search(null, ['alpha'], { scorer, threshold: 1, limit: null })).toEqual([])
    expect(search(null, new Map([['a', 'alpha']]), { scorer, limit: 1 })).toEqual([
      { item: 'alpha', key: 'a', score: 0 },
    ])
    expect(
      search(
        null,
        new Map<string, string | null>([
          ['missing', null],
          ['a', 'alpha'],
          ['b', 'beta'],
        ]),
        { scorer, limit: 2 },
      ),
    ).toEqual([
      { item: 'alpha', key: 'a', score: 0 },
      { item: 'beta', key: 'b', score: 0 },
    ])
    expect(search(null, new Map(), { scorer, limit: null })).toEqual([])
    expect(matcher.best(null, { threshold: 1 })).toBeUndefined()
    expect(matcher.search(null, { limit: 1 })).toEqual([
      { item: 'alpha', key: 0, score: 0 },
    ])
    expect(matcher.search(null, { threshold: 1, limit: null })).toEqual([])
    expect(createMatcher([], { scorer }).best(null)).toBeUndefined()
    expect(matcher.search('alpha', { limit: 0 })).toEqual([])
    expect(matcher.search('alpha', { threshold: 101 })).toEqual([])
  })

  test('cheap arguments are checked before an early exit answers', () => {
    // `limit: 0` and an out-of-bounds threshold are answers, not excuses. A
    // call that names a bare string as its collection, or a threshold that is
    // not a number, is wrong at every limit — the pre-redesign `extract`
    // checked the same three things ahead of its own empty-limit return.
    expect(() =>
      Reflect.apply(search, undefined, [
        'query',
        'not a collection',
        { scorer, limit: 0 },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      search('query', ['alpha'], { scorer, limit: 0, threshold: Number.NaN }),
    ).toThrow(RangeError)
    expect(() =>
      Reflect.apply(bestMatch, undefined, [
        'query',
        'not a collection',
        { scorer, threshold: 101 },
      ]),
    ).toThrow(TypeError)
    expect(() => [
      ...Reflect.apply(searchIter, undefined, [
        'query',
        'not a collection',
        { scorer, threshold: 101 },
      ]),
    ]).toThrow(TypeError)
    const matcher = createMatcher(['alpha'], { scorer })
    expect(() => matcher.search('alpha', { limit: 0, threshold: Number.NaN })).toThrow(
      RangeError,
    )
  })

  test('searchIter reads its call options where the call is made', () => {
    // Lazy scoring, eager arguments: a caller who reuses and mutates an options
    // object must not change a search they already asked for.
    const items = ['alpha', 'alpine', 'beta']
    const options = { scorer, threshold: 90 }
    const iterator = searchIter('alpha', items, options)
    options.threshold = 0
    expect([...iterator].map((match) => match.item)).toEqual(['alpha'])

    const call = { threshold: 90 }
    const fromMatcher = createMatcher(items, { scorer }).searchIter('alpha', call)
    call.threshold = 0
    expect([...fromMatcher].map((match) => match.item)).toEqual(['alpha'])
  })

  test('a finite one-shot heap stops once it holds only optimal scores', () => {
    // What the specialized Matcher drivers already do: ties lose on source
    // order, so a full heap of optimal scores cannot be displaced by anything
    // still ahead. `getText` counts how far the scan actually got.
    let reads = 0
    const counted = (item: string): string => {
      reads++
      return item
    }
    expect(
      search('match', ['match', 'match', 'match', 'match'], {
        scorer,
        getText: counted,
        limit: 2,
      }),
    ).toEqual([
      { item: 'match', key: 0, score: 100 },
      { item: 'match', key: 1, score: 100 },
    ])
    expect(reads).toBe(2)

    reads = 0
    expect(
      search('match', ['zeta', 'yotta', 'match', 'match', 'match'], {
        scorer,
        getText: counted,
        limit: 2,
      }),
    ).toEqual([
      { item: 'match', key: 2, score: 100 },
      { item: 'match', key: 3, score: 100 },
    ])
    expect(reads).toBe(4)

    reads = 0
    const keyed = new Map([
      ['a', 'match'],
      ['b', 'match'],
      ['c', 'match'],
    ])
    expect(search('match', keyed, { scorer, getText: counted, limit: 2 })).toEqual([
      { item: 'match', key: 'a', score: 100 },
      { item: 'match', key: 'b', score: 100 },
    ])
    expect(reads).toBe(2)

    reads = 0
    const displaced = new Map([
      ['a', 'zeta'],
      ['b', 'yotta'],
      ['c', 'match'],
      ['d', 'match'],
      ['e', 'match'],
    ])
    expect(search('match', displaced, { scorer, getText: counted, limit: 2 })).toEqual([
      { item: 'match', key: 'c', score: 100 },
      { item: 'match', key: 'd', score: 100 },
    ])
    expect(reads).toBe(4)
  })

  test('one-shot keyed heaps retain only the best finite result set', () => {
    const numeric = createScorer((_query, choice) => Number(choice), {
      direction: 'similarity',
      bounds: [0, 10],
      symmetric: false,
    })
    const choices = new Map<string, string | null>([
      ['missing', null],
      ['one', '1'],
      ['two', '2'],
      ['three', '3'],
      ['four', '4'],
      ['lower', '3'],
      ['zero', '0'],
    ])
    expect(search('0', choices, { scorer: numeric, limit: 2 })).toEqual([
      { item: '4', key: 'four', score: 4 },
      { item: '3', key: 'three', score: 3 },
    ])
    expect(bestMatch('0', choices, { scorer: numeric, threshold: 3 })).toEqual({
      item: '4',
      key: 'four',
      score: 4,
    })
    expect(bestMatch('same', new Map([['exact', 'same']]), { scorer })).toEqual({
      item: 'same',
      key: 'exact',
      score: 100,
    })
    expect(search('0', ['2', '2', '2'], { scorer: numeric, limit: 2 })).toEqual([
      { item: '2', key: 0, score: 2 },
      { item: '2', key: 1, score: 2 },
    ])
    expect(
      search(
        '0',
        new Map([
          ['low', '1'],
          ['high', '4'],
        ]),
        { scorer: numeric, threshold: 3, limit: 2 },
      ),
    ).toEqual([{ item: '4', key: 'high', score: 4 }])
  })

  test('resolves extraction, normalization, and missing policies once per run', () => {
    const items = [null, { text: null }, { text: ' Alpha ' }]
    const options = {
      scorer,
      getText: (item: { text: string | null } | null) => item?.text,
      normalize: normalizeText,
    }
    expect(bestMatch('alpha', items, options)).toEqual({
      item: items[2],
      key: 2,
      score: 100,
    })
    expect(createMatcher(items, options).best('alpha')?.key).toBe(2)
    expect(() =>
      bestMatch('a', [null], {
        scorer,
        normalize: normalizeText,
        missingItems: 'throw',
      }),
    ).toThrow(TypeError)
    expect(() =>
      bestMatch('a', [null], {
        scorer,
        getText: (item: string | null) => item,
        missingItems: 'throw',
      }),
    ).toThrow(TypeError)
    expect(() =>
      bestMatch('a', [{ text: null }], {
        ...options,
        missingItems: 'throw' as const,
      }),
    ).toThrow(TypeError)
    expect(() =>
      bestMatch('a', [null], {
        ...options,
        missingItems: 'throw' as const,
      }),
    ).toThrow(TypeError)
    expect(() =>
      bestMatch('query', [{ text: 'a' }], {
        scorer,
        getText: (item) => item.text,
        normalize: (value) => (value === 'query' ? value : null),
      }),
    ).toThrow(TypeError)
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
    // A record collection is read with Object.keys, so an object that holds its
    // items anywhere else is a wrong argument rather than an empty collection.
    expect(() =>
      Reflect.apply(createMatcher, undefined, [new Date(), { scorer }]),
    ).toThrow(TypeError)
    const bare: Record<string, string> = Object.create(null)
    bare['only'] = 'alpha'
    expect(createMatcher(bare, { scorer }).best('alpha')?.key).toBe('only')
    // A misspelled policy is refused rather than read as 'throw' by failing
    // the 'skip' test, which is what an untyped caller would otherwise get.
    expect(() =>
      Reflect.apply(createMatcher, undefined, [['a'], { scorer, missingItems: 'none' }]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(bestMatch, undefined, ['a', ['a'], { scorer, missingItems: 'none' }]),
    ).toThrow(TypeError)
    const distance = createScorer(levenshtein.distance)
    expect(() => createMatcher([], { scorer: distance }).best(null)).toThrow(TypeError)
    expect(() => bestMatch(null, [], { scorer: distance })).toThrow(TypeError)
  })
})
