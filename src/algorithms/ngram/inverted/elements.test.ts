// The index accepts what the exhaustive scorer accepts. Every case here is one
// question: does an arbitrary-element corpus answer identically once its
// elements are keyed by ordinal rather than by value?
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import {
  exhaustive,
  exhaustiveScan,
  indexOf,
  LIMITS,
  pairs,
  REPRESENTATION_SPECS,
  THRESHOLDS,
  TVERSKY_SPECS,
  type MetricSpec,
} from '../../../../testing/invertedIndex.js'

const OBJECT_A = { name: 'react' }
const OBJECT_B = { name: 'vue' }
const OBJECT_A_CLONE = { name: 'react' }
const SYMBOL_A = Symbol('a')

const CORPORA: readonly (readonly unknown[][])[] = [
  [
    ['google', 'deepmind', 'ag'],
    ['google', 'ag'],
    ['meta', 'ag'],
  ],
  [['react', 'react'], ['react', 'react', 'react', 'react'], ['react']],
  [
    ['react', 42, true, null],
    [42, true, null, undefined],
    ['react', 42],
  ],
  [[OBJECT_A, OBJECT_B], [OBJECT_A], [OBJECT_A_CLONE], [SYMBOL_A, OBJECT_A]],
  [
    [NaN, 'react'],
    ['react', 'react'],
    [NaN, NaN],
  ],
  [
    [-1, 2, 3],
    [1, 2, 3],
    ['x', 'yy', 'zzz'],
  ],
  // A one-character token and its code point are the same element after
  // conversion, so the two spellings have to agree.
  [
    ['a', 'b', 'c'],
    [97, 98, 99],
    ['ab', 'c'],
  ],
  [
    ['senior', 'software', 'engineer', 'typescript'],
    ['frontend', 'engineer', 'react'],
    ['senior', 'engineer'],
  ],
]

const QUERIES: readonly unknown[][] = [
  [],
  ['google', 'ag'],
  ['react', 'react', 'react'],
  ['react', 42, true, null],
  [OBJECT_A],
  [OBJECT_A_CLONE],
  [SYMBOL_A, OBJECT_A],
  [NaN, 'react'],
  [-1, 2, 3],
  ['a', 'b', 'c'],
  [97, 98, 99],
  ['senior', 'engineer'],
  ['unknown', 'tokens', 'entirely'],
]

const SPECS: readonly MetricSpec[] = [
  { metric: 'dice' },
  { metric: 'cosine' },
  ...TVERSKY_SPECS,
]

describe('an index over arbitrary elements', () => {
  it('answers what the exhaustive scorer answers, across the matrix', () => {
    let cases = 0
    for (const spec of SPECS) {
      for (const gramSize of [1, 2, 3, 4]) {
        for (const choices of CORPORA) {
          const index = indexOf(spec, gramSize, choices)
          for (const query of QUERIES) {
            for (const threshold of THRESHOLDS) {
              for (const limit of LIMITS) {
                expect(pairs(index.select(query, threshold, limit))).toEqual(
                  exhaustive(spec, gramSize, choices, query, threshold, limit),
                )
                cases++
              }
              expect(pairs(index.scan(query, threshold))).toEqual(
                exhaustiveScan(spec, gramSize, choices, query, threshold),
              )
              cases++
            }
          }
        }
      }
    }
    expect(cases).toBeGreaterThan(9000)
  })

  it('matches on randomised element sequences', () => {
    // Object identities come from a fixed pool rather than a generator: three
    // of them cover reference equality and a structural twin, and a generated
    // object would shrink to something nobody can read.
    const element = fc.constantFrom<unknown>(
      'react',
      'typescript',
      'ag',
      'senior',
      'a',
      97,
      0,
      -0,
      1,
      -1,
      1.5,
      true,
      false,
      null,
      undefined,
      NaN,
      OBJECT_A,
      OBJECT_B,
      OBJECT_A_CLONE,
      SYMBOL_A,
    )
    const sequence = fc.array(element, { maxLength: 10 })
    fc.assert(
      fc.property(
        fc.array(sequence, { maxLength: 12 }),
        sequence,
        fc.constantFrom(...THRESHOLDS),
        fc.constantFrom(...LIMITS),
        fc.constantFrom(1, 2, 3, 4),
        fc.constantFrom(...SPECS),
        (choices, query, threshold, limit, gramSize, spec) => {
          const index = indexOf(spec, gramSize, choices)
          expect(pairs(index.select(query, threshold, limit))).toEqual(
            exhaustive(spec, gramSize, choices, query, threshold, limit),
          )
          expect(pairs(index.scan(query, threshold))).toEqual(
            exhaustiveScan(spec, gramSize, choices, query, threshold),
          )
          return true
        },
      ),
      { numRuns: 1000, seed: 0x5eed },
    )
  })

  it('separates identity from structural equality', () => {
    const choices = [[OBJECT_A], [OBJECT_A_CLONE]]
    for (const spec of REPRESENTATION_SPECS) {
      const found = pairs(indexOf(spec, 1, choices).select([OBJECT_A], null, 2))
      expect(found).toEqual(exhaustive(spec, 1, choices, [OBJECT_A], null, 2))
      expect(found[0]).toEqual({ id: 0, score: 1 })
      expect(found[1].score).toBe(0)
    }
  })

  it('leaves NaN matching nothing, itself included', () => {
    // Every Map here keys by SameValueZero, which calls two NaNs equal. The
    // extractor has to overrule that, and the halved score is what says it did.
    const choices = [[NaN, 'react']]
    for (const spec of REPRESENTATION_SPECS) {
      const found = pairs(indexOf(spec, 1, choices).select([NaN, 'react'], null, 1))
      expect(found).toEqual(exhaustive(spec, 1, choices, [NaN, 'react'], null, 1))
      expect(found[0].score).toBeLessThan(1)
    }
    // `react` is shared, the two NaNs are not: one gram of two on each side.
    expect(
      pairs(indexOf({ metric: 'dice' }, 1, choices).select([NaN, 'react'], null, 1))[0]
        .score,
    ).toBe(0.5)
    expect(
      pairs(indexOf({ metric: 'cosine' }, 1, choices).select([NaN, 'react'], null, 1))[0]
        .score,
    ).toBe(0.5)
  })

  it('folds -0 into 0, as the exhaustive profile does', () => {
    const choices = [
      [0, 1],
      [5, 5],
    ]
    for (const spec of REPRESENTATION_SPECS) {
      expect(pairs(indexOf(spec, 1, choices).select([-0, 1], null, 2))).toEqual(
        exhaustive(spec, 1, choices, [-0, 1], null, 2),
      )
    }
  })

  it('indexes token shingles, which is the capability unigrams cannot give', () => {
    const choices = [
      ['senior', 'software', 'engineer', 'typescript'],
      ['software', 'engineer', 'senior', 'typescript'],
    ]
    const spec: MetricSpec = { metric: 'tversky', alpha: 1, beta: 0 }
    const query = ['senior', 'software', 'engineer']
    const found = pairs(indexOf(spec, 2, choices).select(query, null, 2))
    expect(found).toEqual(exhaustive(spec, 2, choices, query, null, 2))
    // Containment of the ordered pair, so the shuffled choice keeps only one
    // of the query's two shingles.
    expect(found[0]).toEqual({ id: 0, score: 1 })
    expect(found[1]).toEqual({ id: 1, score: 0.5 })
  })
})
