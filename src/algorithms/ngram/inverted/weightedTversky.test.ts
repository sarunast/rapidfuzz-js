// The weighted index is an acceleration strategy, so every question here is the
// same one: does it answer what the exhaustive weighted scorer answers?
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import type { ChoiceIndex } from '#core/scoring/choiceIndex.js'
import { createScorer } from '#core/scoring/scorer.js'
import type { Sequence } from '#core/types.js'
import { createIndexedMatcher, createMatcher } from '#search/index.js'

import { similarity as tverskyMetric } from '../../tversky/index.js'
import { compileElementWeights } from '../weightedProfile.js'
import {
  assertWeightGroupsAddressable,
  createWeightedTverskyIndexBuilder,
} from './weightedTversky.js'

interface WeightedSpec {
  readonly alpha?: number
  readonly beta?: number
  readonly weights?: readonly (readonly [unknown, number])[]
  readonly defaultWeight?: number
}

const THRESHOLDS: readonly (number | null)[] = [null, 0, 0.5, 0.8, 1]
const LIMITS: readonly (number | null)[] = [0, 1, 5, null]

function scorerOf(spec: WeightedSpec) {
  return createScorer(tverskyMetric, {
    gramSize: 1,
    ...(spec.alpha === undefined ? {} : { alpha: spec.alpha }),
    ...(spec.beta === undefined ? {} : { beta: spec.beta }),
    ...(spec.weights === undefined ? {} : { elementWeights: new Map(spec.weights) }),
    ...(spec.defaultWeight === undefined
      ? {}
      : { defaultElementWeight: spec.defaultWeight }),
  })
}

function expectParity(
  choices: readonly Sequence[],
  queries: readonly Sequence[],
  spec: WeightedSpec,
): number {
  const scorer = scorerOf(spec)
  const exhaustive = createMatcher(choices, { scorer })
  const indexed = createIndexedMatcher(choices, { scorer })
  let cases = 0
  for (const query of queries) {
    for (const threshold of THRESHOLDS) {
      for (const limit of LIMITS) {
        const call = threshold === null ? { limit } : { limit, threshold }
        expect(indexed.search(query, call)).toEqual(exhaustive.search(query, call))
        cases++
      }
      const call = threshold === null ? undefined : { threshold }
      expect([...indexed.searchIter(query, call)]).toEqual([
        ...exhaustive.searchIter(query, call),
      ])
      expect(indexed.best(query, call)).toEqual(exhaustive.best(query, call))
      cases += 2
    }
  }
  return cases
}

// A common token in more than two thirds of the rows is what makes a posting
// dense; `rare` names one row, `middle` names a third of them.
const DENSE_ROWS: readonly string[][] = [
  ['common', 'middle', 'alpha'],
  ['common', 'middle', 'beta'],
  ['common', 'gamma'],
  ['common', 'middle', 'delta'],
  ['common', 'epsilon'],
  ['common', 'zeta'],
  ['common', 'eta', 'eta'],
  ['rare', 'theta'],
]

const SPARSE_ROWS: readonly string[][] = [
  ['alpha', 'beta'],
  ['gamma', 'delta'],
  ['alpha', 'gamma'],
  ['epsilon'],
  [],
  ['alpha', 'alpha', 'beta'],
  ['zeta', Number.NaN.toString()],
]

const NUMERIC_ROWS: readonly number[][] = [[1, 2, 3], [2, 3, 4], [1, 1, 5], [9], []]

const QUERIES: readonly Sequence[] = [
  ['common', 'middle'],
  ['alpha'],
  ['alpha', 'alpha'],
  ['common'],
  ['unknown'],
  ['rare', 'theta'],
  [],
  ['eta', 'eta'],
]

describe('an indexed weighted search answers what the exhaustive one does', () => {
  it('matches across posting shapes, thresholds and limits', () => {
    const specs: readonly WeightedSpec[] = [
      // Ordinary tiers, and the default weight pair the unweighted metric would
      // have routed to Dice's index.
      {
        weights: [
          ['common', 0.2],
          ['alpha', 3],
          ['middle', 1],
        ],
      },
      {
        alpha: 1,
        beta: 0,
        weights: [
          ['common', 0.2],
          ['alpha', 3],
        ],
      },
      {
        alpha: 1,
        beta: 0.1,
        weights: [
          ['common', 0.05],
          ['alpha', 5],
          ['eta', 2],
        ],
      },
      {
        alpha: 2,
        beta: 10,
        weights: [
          ['common', 0.2],
          ['alpha', 3],
        ],
      },
      // A dense token weighing nothing: it must not force a corpus scan, and it
      // must not change a score either.
      { weights: [['common', 0]] },
      // A dense token weighing very little: the real dense weighted cost.
      {
        weights: [
          ['common', 0.01],
          ['middle', 0.5],
        ],
      },
      // Only the named vocabulary counts.
      {
        defaultWeight: 0,
        weights: [
          ['alpha', 1],
          ['middle', 2],
        ],
      },
      // Every element ignored: the zero-mass branch, over a whole corpus.
      { defaultWeight: 0 },
    ]
    let cases = 0
    for (const spec of specs) {
      cases += expectParity(DENSE_ROWS, QUERIES, spec)
      cases += expectParity(SPARSE_ROWS, QUERIES, spec)
    }
    expect(cases).toBeGreaterThan(500)
  })

  it('matches over a direct numeric index', () => {
    // Small integers key the postings directly, with no ordinal table at all.
    const queries: readonly Sequence[] = [[1, 2], [1, 1], [9], [7], []]
    expectParity(NUMERIC_ROWS, queries, {
      weights: [
        [1, 3],
        [2, 0.5],
      ],
    })
    // A query element no direct index could hold, next to one it can.
    expectParity(NUMERIC_ROWS, [[1, -4], [1.5], [1, 2 ** 40]], {
      weights: [[1, 3]],
    })
  })

  it('matches over mixed and arbitrary elements', () => {
    const token = Symbol('token')
    const object = { name: 'object' }
    const rows: readonly Sequence[] = [
      [token, 'text', 1],
      [object, 'text'],
      [token, object],
      [Number.NaN, 'text'],
      ['text', 'text'],
    ]
    const queries: readonly Sequence[] = [
      [token, 'text'],
      [object],
      [Number.NaN],
      ['text', 'text'],
      [Symbol('other')],
    ]
    expectParity(rows, queries, {
      weights: [
        [token, 4],
        [object, 2],
        ['text', 0.5],
      ],
    })
  })

  it('matches on randomised token corpora', () => {
    const vocabulary = ['aa', 'bb', 'cc', 'dd', 'ee'] as const
    const row = fc.array(fc.constantFrom(...vocabulary), { maxLength: 5 })
    const weightValues = fc.tuple(
      fc.constantFrom(0, 0.1, 1, 3),
      fc.constantFrom(0, 0.5, 2),
      fc.constantFrom(0, 1, 5),
    )
    fc.assert(
      fc.property(
        fc.array(row, { minLength: 1, maxLength: 10 }),
        row,
        weightValues,
        fc.constantFrom(...THRESHOLDS),
        fc.constantFrom(...LIMITS),
        fc.constantFrom([0.5, 0.5], [1, 0], [1, 0.1], [2, 10]),
        (choices, query, values, threshold, limit, [alpha, beta]) => {
          const scorer = createScorer(tverskyMetric, {
            gramSize: 1,
            alpha,
            beta,
            elementWeights: new Map<unknown, number>([
              ['aa', values[0]],
              ['bb', values[1]],
              ['cc', values[2]],
            ]),
          })
          const exhaustive = createMatcher(choices, { scorer })
          const indexed = createIndexedMatcher(choices, { scorer })
          const call = threshold === null ? { limit } : { limit, threshold }
          expect(indexed.search(query, call)).toEqual(exhaustive.search(query, call))
          expect([...indexed.searchIter(query)]).toEqual([
            ...exhaustive.searchIter(query),
          ])
          return true
        },
      ),
      { numRuns: 300, seed: 0x5eed },
    )
  })
})

describe('weighted thresholds at the exhaustive score', () => {
  // Positive finite doubles order like their bit patterns, so stepping the
  // pattern is stepping one ulp.
  const ULP_VIEW = new DataView(new ArrayBuffer(8))

  function ulpAbove(value: number): number {
    ULP_VIEW.setFloat64(0, value)
    ULP_VIEW.setBigUint64(0, ULP_VIEW.getBigUint64(0) + 1n)
    return ULP_VIEW.getFloat64(0)
  }

  function ulpBelow(value: number): number {
    ULP_VIEW.setFloat64(0, value)
    ULP_VIEW.setBigUint64(0, ULP_VIEW.getBigUint64(0) - 1n)
    return ULP_VIEW.getFloat64(0)
  }

  it('accepts and refuses exactly what the exhaustive scorer does', () => {
    // Weights whose sums round in every direction, over a corpus with a dense
    // posting, so the indexed and exhaustive folds must agree bit for bit.
    const weights: readonly (readonly [unknown, number])[] = [
      ['common', 0.1],
      ['alpha', 0.2],
      ['beta', Number.EPSILON],
      ['gamma', 1],
      ['delta', 1e16],
    ]
    for (const [alpha, beta] of [
      [0.5, 0.5],
      [1, 0.1],
      [1e16, 1],
    ]) {
      const scorer = createScorer(tverskyMetric, {
        gramSize: 1,
        alpha,
        beta,
        elementWeights: new Map(weights),
      })
      const exhaustive = createMatcher(DENSE_ROWS, { scorer })
      const indexed = createIndexedMatcher(DENSE_ROWS, { scorer })
      for (const query of [
        ['common', 'alpha'],
        ['common', 'beta', 'gamma'],
        ['delta', 'common'],
        ['alpha', 'beta', 'delta'],
      ]) {
        const best = exhaustive.best(query)
        if (best === undefined) continue
        for (const threshold of [
          best.score,
          ulpAbove(best.score),
          ulpBelow(best.score),
        ]) {
          const call = { limit: 5, threshold }
          expect(indexed.search(query, call)).toEqual(exhaustive.search(query, call))
        }
      }
    }
  })
})

describe('the weighted index itself', () => {
  function indexOf(
    choices: readonly Sequence[],
    entries: readonly (readonly [unknown, number])[],
    fallback?: number,
  ): ChoiceIndex {
    const weights = compileElementWeights(new Map(entries), fallback)
    const builder = createWeightedTverskyIndexBuilder(
      1,
      0.1,
      weights.groupWeights,
      weights.groupOf,
      weights.defaultGroup,
    )
    for (const choice of choices) builder.add(choice)
    return builder.seal()
  }

  function pairs(selected: {
    ids: Uint32Array
    scores: Float64Array
    length: number
  }): { id: number; score: number }[] {
    const out: { id: number; score: number }[] = []
    for (let at = 0; at < selected.length; at++) {
      out.push({ id: selected.ids[at], score: selected.scores[at] })
    }
    return out
  }

  it('answers nothing when a caller asks for nothing', () => {
    // `limit: 0` is a supported answer rather than an excuse, and the dense
    // corpus is what reaches it with candidates in hand rather than none.
    const index = indexOf(DENSE_ROWS, [
      ['common', 0.2],
      ['alpha', 3],
    ])
    for (const threshold of THRESHOLDS) {
      expect(pairs(index.select(['common', 'alpha'], threshold, 0))).toEqual([])
    }
  })

  it('refuses to be added to or sealed twice', () => {
    const weights = compileElementWeights(new Map([['alpha', 2]]), undefined)
    const builder = createWeightedTverskyIndexBuilder(
      1,
      0.1,
      weights.groupWeights,
      weights.groupOf,
      weights.defaultGroup,
    )
    builder.add(['alpha'])
    builder.seal()
    expect(() => builder.add(['beta'])).toThrow('this index is already sealed')
    expect(() => builder.seal()).toThrow('this index is already sealed')
  })

  it('releases share entries a large query grew, and stays exact after', () => {
    // Three postings of a third of the corpus each: long enough to pass the
    // retained-entry cap without any of them qualifying as dense.
    const size = 99_000
    const tokens = ['aa', 'bb', 'cc']
    const choices = Array.from({ length: size }, (_, id) => [tokens[id % 3], `own${id}`])
    const index = indexOf(choices, [
      ['aa', 2],
      ['bb', 3],
      ['cc', 4],
    ])
    const wide = index.select(tokens, null, 3)
    expect(wide.length).toBe(3)
    // The next modest query answers from freshly sized scratch.
    const narrow = pairs(index.select(['aa', 'own0'], null, 2))
    expect(narrow[0].id).toBe(0)
    expect(narrow[0].score).toBe(1)
  })

  it('keys a corpus no radix can pack, and answers the same', () => {
    // A negative element cannot be a digit at any rung, so the whole index falls
    // back to comma-joined keys — the spelling `elementKey` has to agree with.
    const choices: readonly Sequence[] = [
      [-1, 2],
      [2, 3],
      [-1, -1, 4],
    ]
    const scorer = createScorer(tverskyMetric, {
      gramSize: 1,
      elementWeights: new Map<unknown, number>([
        [-1, 3],
        [2, 0.5],
      ]),
    })
    const exhaustive = createMatcher(choices, { scorer })
    const indexed = createIndexedMatcher(choices, { scorer })
    for (const query of [[-1], [-1, 2], [2], [-5], [1.5], [2 ** 40]]) {
      expect(indexed.search(query, { limit: 3 })).toEqual(
        exhaustive.search(query, { limit: 3 }),
      )
    }
  })

  it('answers a query element no packed index could hold', () => {
    // The corpus packs at the narrowest rung, so these three miss every posting
    // while still costing the query what they weigh.
    const choices: readonly Sequence[] = [
      [1, 2],
      [2, 3],
    ]
    const scorer = createScorer(tverskyMetric, {
      gramSize: 1,
      elementWeights: new Map<unknown, number>([[1, 3]]),
    })
    const exhaustive = createMatcher(choices, { scorer })
    const indexed = createIndexedMatcher(choices, { scorer })
    for (const query of [
      [1, -4],
      [1, 1e9],
      [1, 1.5],
    ]) {
      expect(indexed.search(query, { limit: 2 })).toEqual(
        exhaustive.search(query, { limit: 2 }),
      )
    }
  })
})

describe('the weighted index bounds', () => {
  it('refuses a corpus with more weight group entries than an id can address', () => {
    expect(() => assertWeightGroupsAddressable(0xffff_ffff)).not.toThrow()
    expect(() => assertWeightGroupsAddressable(0x1_0000_0000)).toThrow(
      'cannot exceed 4294967295 weight group entries',
    )
  })

  it('answers repeated queries from the same scratch', () => {
    const scorer = scorerOf({
      weights: [
        ['common', 0.2],
        ['alpha', 3],
      ],
    })
    const exhaustive = createMatcher(DENSE_ROWS, { scorer })
    const indexed = createIndexedMatcher(DENSE_ROWS, { scorer })
    // A dense query, then a sparse one, then dense again: the sparse walk has to
    // see share lists the dense scan left clean.
    for (const query of [
      ['common'],
      ['alpha'],
      ['common', 'alpha'],
      ['unknown'],
      ['common'],
    ]) {
      expect(indexed.search(query, { limit: 3 })).toEqual(
        exhaustive.search(query, { limit: 3 }),
      )
    }
  })
})
