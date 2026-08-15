import { describe, expect, it } from 'vitest'

import { similarity as cosineSimilarity } from '#algorithms/cosine/index.js'
import { similarity as diceSimilarity } from '#algorithms/dice/index.js'
import { similarity as levenshteinSimilarity } from '#algorithms/levenshtein/index.js'

import {
  createIndexedMatcher,
  createMatcher,
  createScorer,
  normalizeText,
} from '../../index.js'
import type { Matcher, MaybeSequence } from '../../index.js'

const METRICS = [
  { name: 'dice', metric: diceSimilarity },
  { name: 'cosine', metric: cosineSimilarity },
] as const

const CHOICES: readonly string[] = [
  'node_modules/react/index.js',
  'node_modules/react-dom/index.js',
  'src/algorithms/dice/index.ts',
  'src/algorithms/cosine/index.ts',
  'src/search/matcher.ts',
  'README.md',
  '',
  'ab',
  'src/algorithms/dice/index.ts',
  '😀 astral/path.ts',
  'a\ud800b',
]

const QUERIES: readonly string[] = [
  'src/algorthms/dice.ts',
  'node_modules/react',
  'README.md',
  '',
  'ab',
  '😀 astral/path.ts',
  'zzzzzzzzzz',
  'index',
]

const THRESHOLDS: readonly (number | undefined)[] = [undefined, 0, 0.3, 0.8, 1]
const LIMITS: readonly (number | null | undefined)[] = [undefined, 1, 3, 100, null]

interface Both<TItem, TKey> {
  readonly exhaustive: Matcher<TItem, TKey, 'similarity'>
  readonly indexed: Matcher<TItem, TKey, 'similarity'>
}

function both(
  metric: (typeof METRICS)[number]['metric'],
  gramSize: number,
  choices: readonly string[],
): Both<string, number> {
  const scorer = createScorer(metric, { gramSize })
  return {
    exhaustive: createMatcher(choices, { scorer }),
    indexed: createIndexedMatcher(choices, { scorer }),
  }
}

/** Every query through both matchers, asserting the pair agrees. */
function agree<TItem, TKey>(
  exhaustive: Matcher<TItem, TKey, 'similarity'>,
  indexed: Matcher<TItem, TKey, 'similarity'>,
): void {
  expect(indexed.size).toBe(exhaustive.size)
  for (const query of QUERIES) {
    expect(indexed.search(query, { limit: null })).toEqual(
      exhaustive.search(query, { limit: null }),
    )
    expect(indexed.best(query)).toEqual(exhaustive.best(query))
  }
}

describe('an indexed matcher answers exactly what an exhaustive one does', () => {
  it('agrees on best, search and searchIter across the whole matrix', () => {
    let cases = 0
    for (const { metric } of METRICS) {
      for (const gramSize of [2, 3]) {
        const { exhaustive, indexed } = both(metric, gramSize, CHOICES)
        expect(indexed.size).toBe(exhaustive.size)
        for (const query of QUERIES) {
          for (const threshold of THRESHOLDS) {
            const call = threshold === undefined ? undefined : { threshold }
            expect(indexed.best(query, call)).toEqual(exhaustive.best(query, call))
            expect([...indexed.searchIter(query, call)]).toEqual([
              ...exhaustive.searchIter(query, call),
            ])
            cases += 2
            for (const limit of LIMITS) {
              const search = limit === undefined ? call : { ...call, limit }
              expect(indexed.search(query, search)).toEqual(
                exhaustive.search(query, search),
              )
              cases++
            }
          }
        }
      }
    }
    expect(cases).toBeGreaterThan(500)
  })

  it('agrees on a Map, whose keys are its own', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 3 })
    const items = new Map(CHOICES.map((choice, index) => [`k${index}`, choice]))
    agree(createMatcher(items, { scorer }), createIndexedMatcher(items, { scorer }))
  })

  it('agrees on a plain object, whose keys are its property names', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 3 })
    const items = Object.fromEntries(
      CHOICES.map((choice, index) => [`k${index}`, choice]),
    )
    agree(createMatcher(items, { scorer }), createIndexedMatcher(items, { scorer }))
  })

  it('agrees on an iterable, whose keys are the positions it yielded', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 3 })
    const items = {
      *[Symbol.iterator]() {
        yield* CHOICES
      },
    }
    agree(createMatcher(items, { scorer }), createIndexedMatcher(items, { scorer }))
  })

  it('agrees when items are objects behind getText', () => {
    const files = CHOICES.map((path, id) => ({ id, path }))
    const scorer = createScorer(diceSimilarity, { gramSize: 3 })
    const getText = (file: { path: string }): string => file.path
    const exhaustive = createMatcher(files, { scorer, getText })
    const indexed = createIndexedMatcher(files, { scorer, getText })
    for (const query of QUERIES) {
      expect(indexed.search(query, { limit: null })).toEqual(
        exhaustive.search(query, { limit: null }),
      )
    }
  })

  it('agrees with a normalizer on both sides', () => {
    const choices = ['Hello World', 'HELLO_WORLD', 'goodbye world']
    const scorer = createScorer(cosineSimilarity, { gramSize: 2 })
    const exhaustive = createMatcher(choices, { scorer, normalize: normalizeText })
    const indexed = createIndexedMatcher(choices, { scorer, normalize: normalizeText })
    for (const query of ['hello world!', 'HELLO WORLD', '']) {
      expect(indexed.search(query, { limit: null })).toEqual(
        exhaustive.search(query, { limit: null }),
      )
      expect(indexed.best(query)).toEqual(exhaustive.best(query))
    }
  })
})

describe('gaps in the collection', () => {
  const gappy: readonly (string | null)[] = [
    'alpha beta',
    null,
    'alpha gamma',
    null,
    'delta',
  ]
  const getText = (item: string | null): MaybeSequence => item

  it('skips them and keeps every other key in place', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    const exhaustive = createMatcher(gappy, { scorer, getText })
    const indexed = createIndexedMatcher(gappy, { scorer, getText })
    expect(indexed.size).toBe(3)
    expect(indexed.size).toBe(exhaustive.size)
    for (const query of ['alpha beta', 'delta', 'zzz']) {
      expect(indexed.search(query, { limit: null })).toEqual(
        exhaustive.search(query, { limit: null }),
      )
      expect(indexed.best(query)).toEqual(exhaustive.best(query))
    }
  })

  it('refuses them under the throwing policy', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    expect(() =>
      createIndexedMatcher(gappy, { scorer, getText, missingItems: 'throw' }),
    ).toThrow(TypeError)
  })

  it('numbers ids by what it kept, not by source position', () => {
    // The gap is what the implicit-id builder exists for: choice `alpha gamma`
    // is source position 2 and index id 1, and its result still says key 2.
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    const indexed = createIndexedMatcher(gappy, { scorer, getText })
    const found = indexed.best('alpha gamma')
    expect(found?.key).toBe(2)
    expect(found?.item).toBe('alpha gamma')
  })
})

describe('what construction borrows', () => {
  it('indexes a choice before the accessor can overwrite it', () => {
    // The indexed reader does not snapshot: the sequence it answers with is
    // whatever the accessor returned, and this accessor returns one buffer
    // every time. Indexing has to happen inside the walk, or every choice is
    // indexed as the last one — here both would match `[7, 8, 9]` and neither
    // `[1, 2, 3]`.
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    const source = [
      [1, 2, 3],
      [7, 8, 9],
    ]
    const reused = [0, 0, 0]
    const getText = (item: readonly number[]): readonly number[] => {
      for (let at = 0; at < item.length; at++) reused[at] = item[at]
      return reused
    }
    const indexed = createIndexedMatcher(source, { scorer, getText })
    // The exhaustive Matcher is the oracle it always is here: its reader owns
    // what it prepares, so the same accessor is safe on that side.
    const exhaustive = createMatcher(source, { scorer, getText })
    for (const query of [
      [1, 2, 3],
      [7, 8, 9],
    ]) {
      expect(indexed.search(query, { limit: null })).toEqual(
        exhaustive.search(query, { limit: null }),
      )
    }
    expect(indexed.best([1, 2, 3])?.key).toBe(0)
  })
})

describe('the generator boundary', () => {
  const scorer = createScorer(diceSimilarity, { gramSize: 2 })

  it('normalizes the query when iteration starts, not when it is asked for', () => {
    const indexed = createIndexedMatcher(['abcd', 'wxyz'], { scorer })
    const exhaustive = createMatcher(['abcd', 'wxyz'], { scorer })
    const query = ['a', 'b', 'c', 'd']
    const indexedIterator = indexed.searchIter(query)
    const exhaustiveIterator = exhaustive.searchIter(query)
    // Mutated after the call and before the first `next()`, which both
    // implementations have to see — the exhaustive one does because
    // `normalizeQuery` sits inside its generator.
    query[0] = 'w'
    query[1] = 'x'
    query[2] = 'y'
    query[3] = 'z'
    expect([...indexedIterator]).toEqual([...exhaustiveIterator])
    expect([...indexed.searchIter(['w', 'x', 'y', 'z'])]).toEqual([
      ...exhaustive.searchIter(['w', 'x', 'y', 'z']),
    ])
  })

  it('borrows nothing before iteration starts', () => {
    const indexed = createIndexedMatcher(['abcd', 'abce', 'zzzz'], { scorer })
    const iterator = indexed.searchIter('abcd')
    // A whole search between asking and draining, while the generator has run
    // no code at all.
    indexed.search('zzzz', { limit: null })
    expect([...iterator]).toEqual([...indexed.searchIter('abcd')])
  })

  it('owns its results before the first yield', () => {
    const indexed = createIndexedMatcher(['abcd', 'abce', 'zzzz'], { scorer })
    const expected = [...indexed.searchIter('abcd')]
    const iterator = indexed.searchIter('abcd')
    const first = iterator.next()
    // The call that would overwrite the index's scratch underneath a live
    // iterator, if the iterator were streaming from it.
    indexed.search('zzzz', { limit: null })
    expect([first.value, ...iterator]).toEqual(expected)
  })
})

describe('what it refuses', () => {
  it('refuses a scorer with no indexed representation', () => {
    const scorer = createScorer(levenshteinSimilarity)
    expect(() => createIndexedMatcher(['a'], { scorer })).toThrow(TypeError)
    expect(() => createIndexedMatcher(['a'], { scorer })).toThrow(
      /no indexed representation/,
    )
  })

  it('names the metrics that do have one', () => {
    const scorer = createScorer(levenshteinSimilarity)
    expect(() => createIndexedMatcher(['a'], { scorer })).toThrow(
      /dice\.similarity and cosine\.similarity/,
    )
  })

  it('has no getPrepared option at all', () => {
    // Refused by the type for a TypeScript caller — naming the key is the
    // error, `undefined` included — and by name for anyone else: a prepared
    // handle is the per-choice representation an index replaces, so it is not
    // an option of this constructor rather than an invalid one. Reached the way
    // every other option-key test reaches one, since the type is the point.
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    expect(() =>
      Reflect.apply(createIndexedMatcher, undefined, [
        ['abcd'],
        { scorer, getPrepared: (item: string) => item },
      ]),
    ).toThrow(/unknown createIndexedMatcher option 'getPrepared'/)
  })

  it('refuses choices whose elements are not integers', () => {
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    expect(() => createIndexedMatcher([[{}, {}, {}]], { scorer })).toThrow(
      /integer elements only/,
    )
  })

  it('refuses an unindexable choice before reading the rest', () => {
    // Construction reads and indexes one choice at a time, so the collection
    // after a bad one is never touched. Collecting the sequences and indexing
    // them afterwards would run every accessor first and refuse the same
    // choice later, which is a different set of side effects.
    const scorer = createScorer(diceSimilarity, { gramSize: 2 })
    let reads = 0
    expect(() =>
      createIndexedMatcher(
        [
          [{}, {}, {}],
          [1, 2, 3],
        ],
        {
          scorer,
          getText: (item: unknown[]) => {
            reads++
            return item
          },
        },
      ),
    ).toThrow(/integer elements only/)
    expect(reads).toBe(1)
  })
})

describe('edges an exhaustive matcher also has', () => {
  const scorer = createScorer(diceSimilarity, { gramSize: 2 })

  it('answers an empty collection', () => {
    const indexed = createIndexedMatcher([], { scorer })
    const exhaustive = createMatcher([], { scorer })
    expect(indexed.size).toBe(0)
    expect(indexed.best('abcd')).toEqual(exhaustive.best('abcd'))
    expect(indexed.best(null)).toEqual(exhaustive.best(null))
    expect(indexed.search('abcd', { limit: null })).toEqual(
      exhaustive.search('abcd', { limit: null }),
    )
    expect([...indexed.searchIter('abcd')]).toEqual([...exhaustive.searchIter('abcd')])
  })

  it('answers a missing query the same way', () => {
    const choices = ['abcd', 'abce']
    const indexed = createIndexedMatcher(choices, { scorer })
    const exhaustive = createMatcher(choices, { scorer })
    for (const query of [null, undefined]) {
      for (const call of [undefined, { threshold: 0 }, { threshold: 0.5 }]) {
        expect(indexed.best(query, call)).toEqual(exhaustive.best(query, call))
        expect([...indexed.searchIter(query, call)]).toEqual([
          ...exhaustive.searchIter(query, call),
        ])
        for (const limit of [1, null, 100]) {
          expect(indexed.search(query, { ...call, limit })).toEqual(
            exhaustive.search(query, { ...call, limit }),
          )
        }
      }
    }
  })

  it('answers a zero limit and an impossible threshold', () => {
    const choices = ['abcd', 'abce']
    const indexed = createIndexedMatcher(choices, { scorer })
    const exhaustive = createMatcher(choices, { scorer })
    expect(indexed.search('abcd', { limit: 0 })).toEqual(
      exhaustive.search('abcd', { limit: 0 }),
    )
    // Past the metric's own bound, so nothing can clear it.
    expect(indexed.search('abcd', { threshold: 1.5, limit: null })).toEqual(
      exhaustive.search('abcd', { threshold: 1.5, limit: null }),
    )
    expect(indexed.best('abcd', { threshold: 1.5 })).toEqual(
      exhaustive.best('abcd', { threshold: 1.5 }),
    )
    expect([...indexed.searchIter('abcd', { threshold: 1.5 })]).toEqual([
      ...exhaustive.searchIter('abcd', { threshold: 1.5 }),
    ])
  })

  it('keeps the scorer it was built from', () => {
    const indexed = createIndexedMatcher(['abcd'], { scorer })
    expect(indexed.scorer).toBe(scorer)
  })

  it('snapshots what it scores', () => {
    const choices = ['abcd', 'abce']
    const indexed = createIndexedMatcher(choices, { scorer })
    const before = indexed.search('abcd', { limit: null })
    choices.push('abcf')
    expect(indexed.search('abcd', { limit: null })).toEqual(before)
  })
})
