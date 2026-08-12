import { describe, expect, expectTypeOf, test, vi } from 'vitest'

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
import type { Match, PreparedChoiceOf, Scorer, Sequence } from '../../src/index.js'
import type { AnyMatcherOptions } from '../../src/search/types.js'
import type { ItemIterable } from '../../src/search/types.js'

describe('one-shot search and Matcher', () => {
  const scorer = createScorer(fuzz.similarity)
  // One object, so preparation and search name the same function: the handle
  // check compares normalizers by identity.
  const normalizing = { normalize: normalizeText }

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
      fuzz.weightedSimilarity,
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
    // The scorer and `missingItems` are read before the exit too: they are what
    // every other limit refuses, and `limit: 0` is a result, not a dialect.
    expect(() =>
      Reflect.apply(search, undefined, [
        'query',
        ['alpha'],
        {
          scorer: { direction: 'similarity', bounds: [0, 100], symmetric: true },
          limit: 0,
        },
      ]),
    ).toThrow('scorer was not created by createScorer')
    expect(() =>
      Reflect.apply(search, undefined, [
        'query',
        ['alpha'],
        { scorer, limit: 0, missingItems: 'ignore' },
      ]),
    ).toThrow("missingItems must be 'skip' or 'throw'")
    // A normalizer is not run for it, though: the exit still costs no work.
    const normalize = vi.fn((value: Sequence) => value)
    expect(search('query', ['alpha'], { scorer, limit: 0, normalize })).toEqual([])
    expect(normalize).not.toHaveBeenCalled()
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

  test('prepared choices score exactly as the text they were prepared from', () => {
    const titles = ['new york mets', 'the wonderful new york mets', 'mets new york', '']
    const rows = titles.map((title) => ({ title, prepared: scorer.prepareChoice(title) }))
    const asText = { scorer, getText: (row: (typeof rows)[number]) => row.title }
    const asPrepared = {
      scorer,
      getPrepared: (row: (typeof rows)[number]) => row.prepared,
    }
    const query = 'new york mets'

    expect(bestMatch(query, rows, asPrepared)).toEqual(bestMatch(query, rows, asText))
    expect(search(query, rows, asPrepared)).toEqual(search(query, rows, asText))
    expect(search(query, rows, { ...asPrepared, limit: 2 })).toEqual(
      search(query, rows, { ...asText, limit: 2 }),
    )
    expect(search(query, rows, { ...asPrepared, limit: null })).toEqual(
      search(query, rows, { ...asText, limit: null }),
    )
    expect([...searchIter(query, rows, asPrepared)]).toEqual([
      ...searchIter(query, rows, asText),
    ])

    // The array `searchIter` scores its first eight candidates without a
    // prepared query; prepared mode has no sequence to score that way, so both
    // sides of that window have to agree.
    const many = Array.from({ length: 20 }, (_, index) => `mets ${index}`)
    const manyRows = many.map((title) => ({
      title,
      prepared: scorer.prepareChoice(title),
    }))
    expect([
      ...searchIter(query, manyRows, {
        scorer,
        getPrepared: (row: (typeof manyRows)[number]) => row.prepared,
      }),
    ]).toEqual([
      ...searchIter(query, manyRows, {
        scorer,
        getText: (row: (typeof manyRows)[number]) => row.title,
      }),
    ])

    const keyed = new Map(rows.map((row, index) => [index, row]))
    expect(bestMatch(query, keyed, asPrepared)).toEqual(bestMatch(query, keyed, asText))
    expect(search(query, keyed, asPrepared)).toEqual(search(query, keyed, asText))
    expect([...searchIter(query, keyed, asPrepared)]).toEqual([
      ...searchIter(query, keyed, asText),
    ])

    const preparedMatcher = createMatcher(rows, asPrepared)
    const textMatcher = createMatcher(rows, asText)
    expect(preparedMatcher.size).toBe(textMatcher.size)
    expect(preparedMatcher.best(query)).toEqual(textMatcher.best(query))
    expect(preparedMatcher.search(query, { limit: null })).toEqual(
      textMatcher.search(query, { limit: null }),
    )
    expect([...preparedMatcher.searchIter(query)]).toEqual([
      ...textMatcher.searchIter(query),
    ])
    const keyedMatcher = createMatcher(keyed, asPrepared)
    expect(keyedMatcher.best(query)).toEqual(createMatcher(keyed, asText).best(query))
  })

  test('a prepared choice is accepted by any scorer that prepares it the same way', () => {
    const rows = [{ prepared: createScorer(fuzz.similarity).prepareChoice('alpha') }]
    const read = (row: (typeof rows)[number]) => row.prepared
    // A second default scorer of the same metric compiles to the same
    // preparation, so it accepts the first one's choices.
    expect(bestMatch('alpha', rows, { scorer, getPrepared: read })?.score).toBe(100)
    let accesses = 0
    const observed = withPublicScoreObserver(scorer, () => {
      accesses++
    })
    expect(bestMatch('alpha', rows, { scorer: observed, getPrepared: read })?.score).toBe(
      100,
    )
    // The clone prepares through the scorer it observes, so its handles are
    // the same scorer's handles.
    expect(
      bestMatch('alpha', [{ prepared: observed.prepareChoice('alpha') }], {
        scorer,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(100)
    expect(accesses).toBe(0)

    // `missing` never reaches preparation, so it does not fork the key; a
    // weighting does, and its choices belong to that scorer alone.
    expect(
      bestMatch('alpha', rows, {
        scorer: createScorer(fuzz.similarity, { missing: 'throw' }),
        getPrepared: read,
      })?.score,
    ).toBe(100)
    const weighted = createScorer(levenshtein.distance, { weights: [1, 2, 1] })
    const weightedRows = [{ prepared: weighted.prepareChoice('alpha') }]
    expect(
      bestMatch('alpha', weightedRows, {
        scorer: weighted,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(0)
    expect(() =>
      bestMatch('alpha', weightedRows, {
        scorer: createScorer(levenshtein.distance),
        getPrepared: (row) => row.prepared,
      }),
    ).toThrow('prepared choice is incompatible with this scorer')
    const configured = weighted
    expect(() =>
      Reflect.apply(bestMatch, undefined, [
        'alpha',
        rows,
        { scorer: configured, getPrepared: read },
      ]),
    ).toThrow('prepared choice is incompatible with this scorer')
    const otherMetric = createScorer(fuzz.tokenSetSimilarity)
    expect(
      () =>
        bestMatch('alpha', rows, {
          scorer: otherMetric,
          getPrepared: () => otherMetric.prepareChoice('alpha'),
        })?.score,
    ).not.toThrow()
    expect(() =>
      Reflect.apply(bestMatch, undefined, [
        'alpha',
        rows,
        { scorer: otherMetric, getPrepared: read },
      ]),
    ).toThrow('prepared choice is incompatible with this scorer')
    // Two scorers configured the same way still prepare for themselves: a
    // configuration is read once, per scorer, and never interned.
    const twin = createScorer(levenshtein.distance, { weights: [1, 2, 1] })
    expect(() =>
      bestMatch('alpha', [{ prepared: configured.prepareChoice('alpha') }], {
        scorer: twin,
        getPrepared: (row) => row.prepared,
      }),
    ).toThrow(TypeError)
  })

  test('getPrepared refuses anything that is not a prepared choice', () => {
    const handle = scorer.prepareChoice('alpha')
    const forged = Object.create(Object.getPrototypeOf(handle))
    for (const value of ['abc', null, undefined, 42, {}, forged, { ...handle }]) {
      const options = { scorer, getPrepared: () => value }
      expect(() => Reflect.apply(bestMatch, undefined, ['a', ['x'], options])).toThrow(
        'getPrepared returned an invalid prepared choice',
      )
      expect(() => Reflect.apply(search, undefined, ['a', ['x'], options])).toThrow(
        'getPrepared returned an invalid prepared choice',
      )
      expect(() => Reflect.apply(createMatcher, undefined, [['x'], options])).toThrow(
        'getPrepared returned an invalid prepared choice',
      )
      // The iterator reports it where it scores, not where it was asked for.
      const stream = Reflect.apply(searchIter, undefined, ['a', ['x'], options])
      expect(() => stream.next()).toThrow(
        'getPrepared returned an invalid prepared choice',
      )
    }
  })

  test('an accessor that is not a function is refused before any item is read', () => {
    // An empty collection never calls one, so the check belongs where the
    // reader is built: otherwise a search over `[]` accepts options that a
    // search over one item would refuse.
    for (const [option, message] of [
      ['getPrepared', 'getPrepared must be a function'],
      ['getText', 'getText must be a function'],
      ['normalize', 'normalize must be a function'],
    ] as const) {
      const options = { scorer, [option]: null }
      for (const entry of [bestMatch, search, searchIter, createMatcher]) {
        const args = entry === createMatcher ? [[], options] : ['a', [], options]
        expect(() => Reflect.apply(entry, undefined, args)).toThrow(message)
      }
    }
    // Prepared mode reads the query through `normalize` and never builds a
    // text reader, so it has to make the same check for itself.
    expect(() =>
      Reflect.apply(bestMatch, undefined, [
        'a',
        [],
        { scorer, getPrepared: () => scorer.prepareChoice('alpha'), normalize: null },
      ]),
    ).toThrow('normalize must be a function')
  })

  test('getPrepared cannot be combined with the text-side options', () => {
    const handle = scorer.prepareChoice('alpha')
    for (const extra of [{ getText: () => 'alpha' }, { missingItems: 'skip' }]) {
      const options = { scorer, getPrepared: () => handle, ...extra }
      for (const entry of [bestMatch, search, searchIter, createMatcher]) {
        const args = entry === createMatcher ? [['x'], options] : ['a', ['x'], options]
        expect(() => Reflect.apply(entry, undefined, args)).toThrow(
          'getPrepared cannot be combined with getText or missingItems',
        )
      }
      expect(() =>
        Reflect.apply(search, undefined, ['a', ['x'], { ...options, limit: 0 }]),
      ).toThrow('getPrepared cannot be combined with getText or missingItems')
    }
    // An option named but left undefined is the shape the types admit, so the
    // runtime accepts it too.
    expect(
      bestMatch('alpha', [{ handle }], {
        scorer,
        getPrepared: (row) => row.handle,
        getText: undefined,
        missingItems: undefined,
      })?.score,
    ).toBe(100)
  })

  test('a prepared search scores handles normalized as its query is', () => {
    const rows = [{ prepared: scorer.prepareChoice('New York Mets!', normalizing) }]
    const read = (row: (typeof rows)[number]) => row.prepared
    // Both sides through the same normalizer, so a query the choice never saw
    // in that spelling still matches it exactly.
    expect(
      bestMatch('NEW YORK METS', rows, { scorer, getPrepared: read, ...normalizing })
        ?.score,
    ).toBe(100)
    expect(
      createMatcher(rows, { scorer, getPrepared: read, ...normalizing }).best(
        'NEW YORK METS',
      )?.score,
    ).toBe(100)
  })

  test('a prepared search refuses a handle normalized unlike its query', () => {
    const plain = [{ prepared: scorer.prepareChoice('New York Mets!') }]
    const normalized = [{ prepared: scorer.prepareChoice('New York Mets!', normalizing) }]
    const read = (row: { prepared: PreparedChoiceOf<typeof scorer> }) => row.prepared
    // The mistake the check exists for: text prepared raw, scored against a
    // query that was normalized, which used to answer a plausible 0.
    expect(() =>
      bestMatch('new york mets', plain, { scorer, getPrepared: read, ...normalizing }),
    ).toThrow('this search normalizes, the prepared choice was not')
    expect(() =>
      bestMatch('new york mets', normalized, { scorer, getPrepared: read }),
    ).toThrow('prepared choice was normalized, this search is not')
    // Identity, not equivalence: a normalizer that does the same thing is
    // still a different function.
    expect(() =>
      bestMatch('new york mets', normalized, {
        scorer,
        getPrepared: read,
        normalize: (value) => normalizeText(value),
      }),
    ).toThrow('prepared choice was normalized by a different function than this search')
    // Every entry point resolves through the same reader.
    for (const entry of [bestMatch, search, createMatcher]) {
      const options = { scorer, getPrepared: read, ...normalizing }
      const args = entry === createMatcher ? [plain, options] : ['a', plain, options]
      expect(() => Reflect.apply(entry, undefined, args)).toThrow(TypeError)
    }
    expect(() =>
      searchIter('a', plain, { scorer, getPrepared: read, ...normalizing }).next(),
    ).toThrow(TypeError)
  })

  test('normalize is read once, so an accessor cannot split the two sides', () => {
    // The check compares the handle's normalizer to the search's, and both used
    // to be read from the options object separately: an accessor answering a
    // different function on a later read satisfied the check and then
    // normalized the query some other way, scoring a pair made differently.
    const identity = (value: Sequence): Sequence => value
    const rows = [{ prepared: scorer.prepareChoice('ZURICH', normalizing) }]
    let reads = 0
    const options = {
      scorer,
      getPrepared: (row: (typeof rows)[number]) => row.prepared,
      get normalize() {
        reads++
        return reads === 1 ? normalizeText : identity
      },
    }
    for (const run of [
      () => bestMatch('ZÜRICH!', rows, options)?.score,
      () => search('ZÜRICH!', rows, options)[0]?.score,
      () => [...searchIter('ZÜRICH!', rows, options)][0]?.score,
      () => createMatcher(rows, options).best('ZÜRICH!')?.score,
    ]) {
      reads = 0
      // The normalized score, not the 0 an unnormalized query would answer.
      expect(run()).toBeCloseTo(83.333, 3)
      expect(reads).toBe(1)
    }
  })

  test('a caller who preprocesses both sides names no normalizer at all', () => {
    // The other supported arrangement: preparation and query are the caller's,
    // the search normalizes nothing, and the two sides still agree.
    const rows = [{ prepared: scorer.prepareChoice(normalizeText('New York Mets!')) }]
    expect(
      bestMatch(normalizeText('NEW YORK METS'), rows, {
        scorer,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(100)
  })

  test('a candidate a guard rejects is never read or checked', () => {
    // A generator decides what is worth scoring, so a handle prepared under
    // another normalizer — or none — costs nothing until it is yielded.
    let reads = 0
    const rows = [
      { keep: false, prepared: scorer.prepareChoice('alpha') },
      { keep: true, prepared: scorer.prepareChoice('Alpha!', normalizing) },
    ]
    function* plausible(): Generator<(typeof rows)[number]> {
      for (const row of rows) {
        if (!row.keep) continue
        yield row
      }
    }
    const matches = [
      ...searchIter('ALPHA', plausible(), {
        scorer,
        getPrepared: (row) => {
          reads++
          return row.prepared
        },
        ...normalizing,
      }),
    ]
    expect(matches.map((match) => match.score)).toEqual([100])
    expect(reads).toBe(1)
  })

  test('a missing query answers presence through the prepared reader', () => {
    const rows = [{ prepared: scorer.prepareChoice('alpha') }]
    const read = (row: (typeof rows)[number]) => row.prepared
    const options = { scorer, getPrepared: read }
    expect(bestMatch(null, rows, options)).toEqual({ item: rows[0], key: 0, score: 0 })
    expect(search(null, rows, options)).toEqual([{ item: rows[0], key: 0, score: 0 }])
    expect([...searchIter(null, rows, options)]).toEqual([
      { item: rows[0], key: 0, score: 0 },
    ])
    const keyed = new Map([['only', rows[0]]])
    expect(bestMatch(null, keyed, options)?.key).toBe('only')
    expect(search(null, keyed, options)).toHaveLength(1)
    expect([...searchIter(null, keyed, options)]).toHaveLength(1)
    // Presence resolves the handle, so a misused one still reports itself.
    const broken = { scorer, getPrepared: () => 'nonsense' }
    expect(() => Reflect.apply(bestMatch, undefined, [null, ['x'], broken])).toThrow(
      TypeError,
    )
    expect(() => Reflect.apply(search, undefined, [null, ['x'], broken])).toThrow(
      TypeError,
    )
    expect(() =>
      Reflect.apply(searchIter, undefined, [null, ['x'], broken]).next(),
    ).toThrow(TypeError)
  })

  test('a prepared Matcher resolves its handles once, at construction', () => {
    let reads = 0
    const rows = ['alpha', 'beta'].map((text) => ({
      prepared: scorer.prepareChoice(text),
    }))
    let accesses = 0
    const observed = withPublicScoreObserver(scorer, () => {
      accesses++
    })
    const matcher = createMatcher(rows, {
      scorer: observed,
      getPrepared: (row) => {
        reads++
        return row.prepared
      },
    })
    expect(reads).toBe(2)
    expect(matcher.best('alpha')?.score).toBe(100)
    expect(matcher.search('alpha', { limit: null })).toHaveLength(2)
    expect([...matcher.searchIter('alpha')]).toHaveLength(2)
    expect(reads).toBe(2)
    expect(accesses).toBe(0)
  })

  test('prepared options infer their key, direction, and brand', () => {
    const rows = [{ prepared: scorer.prepareChoice('alpha') }]
    expectTypeOf(
      bestMatch('a', rows, { scorer, getPrepared: (row) => row.prepared }),
    ).toEqualTypeOf<Match<(typeof rows)[number], number> | undefined>()
    expectTypeOf(
      bestMatch('a', new Map([[Symbol('k'), rows[0]]]), {
        scorer,
        getPrepared: (row) => row.prepared,
      })?.key,
    ).toEqualTypeOf<symbol | undefined>()
    expectTypeOf<PreparedChoiceOf<typeof scorer>>().toEqualTypeOf(
      scorer.prepareChoice('alpha'),
    )

    // Two metrics' handles are different types, which is what stops one
    // reaching the other's scorer. That the search functions refuse the
    // crossing is checked against the built package by `pnpm check:consumer`,
    // where generic inference is the thing under test.
    const distance = createScorer(levenshtein.distance)
    expectTypeOf(distance.prepareChoice('alpha')).not.toExtend<
      PreparedChoiceOf<typeof scorer>
    >()
    expectTypeOf(scorer.prepareChoice('alpha')).not.toExtend<
      PreparedChoiceOf<typeof distance>
    >()
    // Widening the scorer gives the brand up on purpose: the runtime key is
    // what remains.
    const widened: Scorer<'similarity'> = scorer
    expectTypeOf(scorer.prepareChoice('alpha')).toExtend<
      PreparedChoiceOf<typeof widened>
    >()
    // An options literal carrying both accessors satisfies neither member.
    type Row = { prepared: PreparedChoiceOf<typeof scorer> }
    expectTypeOf<{
      scorer: typeof scorer
      getPrepared: (row: Row) => PreparedChoiceOf<typeof scorer>
      getText: (row: Row) => string
    }>().not.toExtend<AnyMatcherOptions<Row, 'similarity'>>()
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

  test('an option a call does not define is refused rather than ignored', () => {
    // The defect this exists for: a misspelled optional key leaves the call
    // typechecking — freshness is lost the moment the object is a variable —
    // and silently turns the behaviour it names off.
    const misspelled = { scorer, thresold: 90 }
    for (const [entry, label] of [
      [bestMatch, 'bestMatch'],
      [search, 'search'],
      [searchIter, 'searchIter'],
      [createMatcher, 'createMatcher'],
    ] as const) {
      const args =
        entry === createMatcher ? [['x'], misspelled] : ['a', ['x'], misspelled]
      expect(() => Reflect.apply(entry, undefined, args)).toThrow(
        `unknown ${label} option 'thresold'`,
      )
      // Refused where the call is made, not where it is iterated: the shape of
      // the call is settled before any scoring.
      const bad = entry === createMatcher ? [['x'], null] : ['a', ['x'], null]
      expect(() => Reflect.apply(entry, undefined, bad)).toThrow(
        `${label} options must be an object`,
      )
    }
    // Each entry point takes the keys it defines. `limit` is `search`'s, so a
    // bag carrying it is not shared with the two that would ignore it.
    expect(() =>
      Reflect.apply(bestMatch, undefined, ['a', ['x'], { scorer, limit: 1 }]),
    ).toThrow("unknown bestMatch option 'limit'")
    expect(() =>
      Reflect.apply(searchIter, undefined, ['a', ['x'], { scorer, limit: 1 }]),
    ).toThrow("unknown searchIter option 'limit'")
    // Named but left undefined is still a misspelling, not an absent option.
    expect(() =>
      Reflect.apply(search, undefined, ['a', ['x'], { scorer, limti: undefined }]),
    ).toThrow("unknown search option 'limti'")
  })

  test('a Matcher method takes the threshold and the limit, and no more', () => {
    const matcher = createMatcher(['alpha'], { scorer })
    // A scorer named here would be ignored — the Matcher scores with the one it
    // was built from — so it is refused instead.
    expect(() => Reflect.apply(matcher.best, matcher, ['alpha', { scorer }])).toThrow(
      "unknown matcher.best option 'scorer'",
    )
    expect(() =>
      Reflect.apply(matcher.searchIter, matcher, ['alpha', { threshold: 1, limit: 1 }]),
    ).toThrow("unknown matcher.searchIter option 'limit'")
    expect(() =>
      Reflect.apply(matcher.search, matcher, ['alpha', { thresold: 1 }]),
    ).toThrow("unknown matcher.search option 'thresold'")
    for (const method of [matcher.best, matcher.search, matcher.searchIter]) {
      expect(() => Reflect.apply(method, matcher, ['alpha', null])).toThrow(
        'options must be an object',
      )
    }
    // No options at all is the common call, and has no keys to walk.
    expect(matcher.best('alpha')?.score).toBe(100)
    expect(matcher.search('alpha', { limit: 1, threshold: 90 })).toHaveLength(1)
    expect(Array.from(matcher.searchIter('alpha', { threshold: 90 }))).toHaveLength(1)
  })

  test('search at limit 1 keeps its own option list', () => {
    // It answers through the same scan `bestMatch` runs, which must not mean
    // being checked a second time against a list without `limit` in it.
    expect(search('alpha', ['beta', 'alpha'], { scorer, limit: 1 })).toEqual([
      { item: 'alpha', key: 1, score: 100 },
    ])
    const match = bestMatch('alpha', ['beta', 'alpha'], { scorer })
    expect(search('alpha', ['beta', 'alpha'], { scorer, limit: 1 })[0]).toEqual(match)
    // And the checks it delegates past still run in the same order.
    expect(() =>
      Reflect.apply(search, undefined, ['a', 'not a collection', { scorer, limit: 1 }]),
    ).toThrow(TypeError)
  })
})
