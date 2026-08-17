// `explain` at `gramSize: 1`: the occurrence-level evidence behind a score, and
// the contract that it agrees with every other path to the bit.

import fc from 'fast-check'
import { describe, expect, expectTypeOf, it } from 'vitest'

import { createScorer, type Scorer, type ScorerOf } from '#core/scoring/scorer.js'
import { createIndexedMatcher, createMatcher } from '#search/index.js'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { callUntyped } from '../../../testing/untyped.js'
import { similarity as levenshteinSimilarity } from '../levenshtein/index.js'
import type { TverskyEvidence } from './evidence.js'
import { tverskyDistance, tverskySimilarity } from './implementation.js'
import {
  distance as tverskyDistanceMetric,
  similarity as tverskyMetric,
  type TverskyExplainConfiguration,
  type TverskySimilarityExplainConfiguration,
} from './index.js'

type Elements = readonly unknown[] | string

interface Configuration {
  readonly alpha?: number
  readonly beta?: number
  readonly elementWeights?: ReadonlyMap<unknown, number>
  readonly defaultElementWeight?: number
}

function unigram(
  configuration: Configuration,
): Configuration & { readonly gramSize: 1 } & Readonly<Record<string, unknown>> {
  return { ...configuration, gramSize: 1 }
}

function explainer(configuration: Configuration) {
  return createScorer(tverskyMetric, unigram(configuration))
}

function distanceExplainer(configuration: Configuration) {
  return createScorer(tverskyDistanceMetric, unigram(configuration))
}

/**
 * Every path a similarity has, including evidence. The index takes only a
 * similarity scorer by design, so the distance matrix below is one arm shorter.
 */
function everyPath(a: Elements, b: Elements, configuration: Configuration): number {
  const options = unigram(configuration)
  const scorer = explainer(configuration)
  const score = tverskySimilarity(a, b, options)
  expect(scorer.score(a, b)).toBe(score)
  expect(prepareScorerOf(tverskySimilarity)(a, options)(b, null)).toBe(score)
  // No `?? 0` fallback: a legitimate score of 0 would then be indistinguishable
  // from a matcher wrongly returning nothing at all.
  expect(createMatcher([b], { scorer }).best(a)?.score).toBe(score)
  expect(createIndexedMatcher([b], { scorer }).best(a)?.score).toBe(score)
  expect(scorer.explain(a, b).score).toBe(score)
  expect(scorer.explain(a, b).similarity).toBe(score)
  return score
}

/** The distance side: the same paths minus the index, which refuses a distance. */
function everyDistancePath(
  a: Elements,
  b: Elements,
  configuration: Configuration,
): number {
  const options = unigram(configuration)
  const scorer = distanceExplainer(configuration)
  const score = tverskyDistance(a, b, options)
  expect(scorer.score(a, b)).toBe(score)
  expect(prepareScorerOf(tverskyDistance)(a, options)(b, null)).toBe(score)
  expect(createMatcher([b], { scorer }).best(a)?.score).toBe(score)
  expect(scorer.explain(a, b).score).toBe(score)
  return score
}

function weightsFor(
  entries: readonly (readonly [unknown, number])[],
): Map<unknown, number> {
  return new Map(entries)
}

const COMPANY = weightsFor([
  ['swisscom', 5],
  ['google', 4],
  ['ag', 0.1],
  ['gmbh', 0.1],
])

describe('evidence agrees with the score on every path', () => {
  const configurations: readonly Configuration[] = [
    {},
    { alpha: 1, beta: 0 },
    { alpha: 2, beta: 10 },
    { elementWeights: COMPANY },
    { elementWeights: COMPANY, alpha: 1, beta: 0.1 },
    { elementWeights: weightsFor([['ag', 0]]) },
    { defaultElementWeight: 0, elementWeights: weightsFor([['google', 5]]) },
    { defaultElementWeight: 7 },
  ]

  it('reports the scorer score, bit for bit, in both directions', () => {
    const pairs: readonly (readonly [Elements, Elements])[] = [
      [['swisscom', 'ag'], ['swisscom']],
      [
        ['google', 'gmbh'],
        ['google', 'ag'],
      ],
      [['ag'], ['gmbh']],
      [[], ['x']],
      [[], []],
      [['react', 'react'], ['react']],
    ]
    for (const configuration of configurations) {
      for (const [a, b] of pairs) {
        expect(everyDistancePath(a, b, configuration)).toBe(
          1 - everyPath(a, b, configuration),
        )
      }
    }
  })

  it('reports it for arbitrary unigram inputs', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom('swisscom', 'google', 'ag', 'gmbh'), { maxLength: 6 }),
        fc.array(fc.constantFrom('swisscom', 'google', 'ag', 'gmbh'), { maxLength: 6 }),
        fc.constantFrom(...configurations),
        (a, b, configuration) => {
          everyPath(a, b, configuration)
        },
      ),
    )
  })

  it('inverts to the distance the distance scorer reports', () => {
    const scorer = distanceExplainer({ elementWeights: COMPANY })
    const evidence = scorer.explain(['swisscom', 'ag'], ['swisscom'])
    expect(evidence.score).toBe(1 - evidence.similarity)
    expect(evidence.score).toBe(scorer.score(['swisscom', 'ag'], ['swisscom']))
  })
})

describe('the totals a score is made of', () => {
  it('reproduces the score from the three components alone', () => {
    const scorer = explainer({ elementWeights: COMPANY, alpha: 1, beta: 0.1 })
    const evidence = scorer.explain(['swisscom', 'ag'], ['swisscom'])
    const { sharedMass, firstUnmatchedMass, secondUnmatchedMass } = evidence.totals
    expect(
      sharedMass / (sharedMass + 1 * firstUnmatchedMass + 0.1 * secondUnmatchedMass),
    ).toBe(evidence.similarity)
  })

  it('folds each side independently rather than deriving one from another', () => {
    const evidence = explainer({ elementWeights: COMPANY }).explain(
      ['swisscom', 'ag'],
      ['swisscom'],
    )
    expect(evidence.totals).toEqual({
      firstMass: 5.1,
      secondMass: 5,
      sharedMass: 5,
      firstUnmatchedMass: 0.1,
      secondUnmatchedMass: 0,
    })
  })

  it('counts occurrences where no weighting applies', () => {
    const evidence = explainer({}).explain(['a', 'b'], ['b', 'c'])
    expect(evidence.totals).toEqual({
      firstMass: 2,
      secondMass: 2,
      sharedMass: 1,
      firstUnmatchedMass: 1,
      secondUnmatchedMass: 1,
    })
  })

  it('keeps a tiny remainder a rounded mass would swallow', () => {
    const elementWeights = weightsFor([
      ['x', 1e16],
      ['y', 1],
    ])
    const evidence = explainer({ elementWeights, alpha: 1e16 }).explain(['x', 'y'], ['x'])
    expect(evidence.totals.sharedMass).toBe(1e16)
    expect(evidence.totals.firstUnmatchedMass).toBe(1)
    expect(evidence.totals.secondUnmatchedMass).toBe(0)
  })

  // A second weight in each table keeps the weighting genuinely weighted: one
  // amount everywhere is uniform-positive, which compiles the table away and
  // would explain at weight 1 through the unweighted engine instead.
  it('survives an extreme weight on its own, in either magnitude', () => {
    const huge = explainer({
      elementWeights: weightsFor([
        ['x', Number.MAX_VALUE],
        ['unused', Number.MAX_VALUE / 2],
      ]),
    }).explain(['x'], ['x'])
    expect(huge.similarity).toBe(1)
    expect(huge.totals.sharedMass).toBeGreaterThan(0)
    expect(huge.matches[0]?.sharedMass).toBe(huge.totals.sharedMass)

    const tiny = explainer({
      elementWeights: weightsFor([
        ['x', Number.MIN_VALUE],
        ['unused', 2 * Number.MIN_VALUE],
      ]),
      defaultElementWeight: Number.MIN_VALUE,
    }).explain(['x'], ['x'])
    expect(tiny.similarity).toBe(1)
    expect(tiny.totals.sharedMass).toBe(Number.MIN_VALUE)
    expect(tiny.matches[0]?.sharedMass).toBe(Number.MIN_VALUE)
  })

  for (const weight of [Number.EPSILON, 1e-300]) {
    it(`keeps a remainder of ${weight} on the side that carries it`, () => {
      const elementWeights = weightsFor([
        ['x', 1],
        ['y', weight],
      ])
      const evidence = explainer({ elementWeights }).explain(['x', 'y'], ['x', 'y'])
      expect(evidence.totals.sharedMass).toBe(1 + weight)
      expect(evidence.matches[1]?.sharedMass).toBe(weight)

      const dropped = explainer({ elementWeights }).explain(['x', 'y'], ['x'])
      expect(dropped.totals.firstUnmatchedMass).toBe(weight)
      expect(dropped.unmatchedFirst[0]?.unmatchedMass).toBe(weight)
    })
  }

  it('still refuses a span it cannot rescale losslessly', () => {
    expect(() =>
      explainer({
        elementWeights: weightsFor([
          ['x', Number.MAX_VALUE],
          ['y', Number.MIN_VALUE],
        ]),
      }),
    ).toThrow(RangeError)
  })
})

// Score parity alone cannot check the decomposition: the scorer and evidence
// both call `weightedComponents`, so they would agree even if it were wrong.
// Expanding each element by an integer weight turns a weighted comparison into
// an unweighted one over a larger multiset, which is an independent model of
// what the five masses mean.
describe('the integer-weight expansion oracle', () => {
  const vocabulary = ['aa', 'bb', 'cc'] as const

  function expand(
    values: readonly string[],
    weights: ReadonlyMap<unknown, number>,
  ): string[] {
    const expanded: string[] = []
    for (const element of values) {
      const weight = weights.get(element) ?? 1
      for (let at = 0; at < weight; at++) expanded.push(element)
    }
    return expanded
  }

  it('decomposes into the masses of the expanded unweighted multiset', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...vocabulary), { maxLength: 5 }),
        fc.array(fc.constantFrom(...vocabulary), { maxLength: 5 }),
        fc.tuple(
          fc.integer({ min: 1, max: 4 }),
          fc.integer({ min: 1, max: 4 }),
          fc.integer({ min: 1, max: 4 }),
        ),
        fc.tuple(fc.constantFrom(0, 0.5, 1, 2), fc.constantFrom(0, 0.5, 1, 2)),
        (a, b, values, [alpha, beta]) => {
          if (alpha === 0 && beta === 0) return
          // `defaultElementWeight` stays 1, so the only uniform case is every
          // weight being 1 — where expanding by 1 is the identity anyway.
          const elementWeights = weightsFor(
            vocabulary.map((element, index) => [element, values[index]]),
          )
          const weighted = explainer({ elementWeights, alpha, beta }).explain(a, b)
          const expanded = explainer({ alpha, beta }).explain(
            expand(a, elementWeights),
            expand(b, elementWeights),
          )
          expect(weighted.totals).toEqual(expanded.totals)
          expect(weighted.similarity).toBe(expanded.similarity)
        },
      ),
    )
  })
})

describe('degenerate totals stay consistent with the occurrences', () => {
  it('gives an empty side no mass and the other side all of its own', () => {
    const evidence = explainer({}).explain([], ['x'])
    expect(evidence.totals).toEqual({
      firstMass: 0,
      secondMass: 1,
      sharedMass: 0,
      firstUnmatchedMass: 0,
      secondUnmatchedMass: 1,
    })
    expect(evidence.unmatchedSecond).toEqual([
      { element: 'x', index: 0, weight: 1, unmatchedMass: 1 },
    ])
  })

  it('keeps the priced side priced when the other carries no mass at all', () => {
    const scorer = explainer({
      defaultElementWeight: 0,
      elementWeights: weightsFor([['google', 5]]),
    })
    const evidence = scorer.explain(['google'], ['ag'])
    expect(evidence.similarity).toBe(0)
    expect(evidence.totals).toEqual({
      firstMass: 5,
      secondMass: 0,
      sharedMass: 0,
      firstUnmatchedMass: 5,
      secondUnmatchedMass: 0,
    })
    expect(evidence.unmatchedFirst).toEqual([
      { element: 'google', index: 0, weight: 5, unmatchedMass: 5 },
    ])
    expect(evidence.unmatchedSecond).toEqual([])
  })

  it('reports the zero-mass equality rule as empty evidence', () => {
    const scorer = explainer({ defaultElementWeight: 0 })
    const equal = scorer.explain(['ag'], ['ag'])
    const different = scorer.explain(['ag'], ['gmbh'])
    const unmatchable = scorer.explain([Number.NaN], [Number.NaN])

    for (const evidence of [equal, different, unmatchable]) {
      expect(evidence.matches).toEqual([])
      expect(evidence.unmatchedFirst).toEqual([])
      expect(evidence.unmatchedSecond).toEqual([])
      expect(evidence.totals).toEqual({
        firstMass: 0,
        secondMass: 0,
        sharedMass: 0,
        firstUnmatchedMass: 0,
        secondUnmatchedMass: 0,
      })
    }
    expect(equal.similarity).toBe(1)
    expect(different.similarity).toBe(0)
    expect(unmatchable.similarity).toBe(0)
  })

  it('agrees with the scorer on every zero-mass shape', () => {
    const ignored = { defaultElementWeight: 0 }
    expect(everyPath(['ag'], ['ag'], ignored)).toBe(1)
    expect(everyPath(['ag', 'gmbh'], ['gmbh', 'ag'], ignored)).toBe(1)
    expect(everyPath(['ag'], ['gmbh'], ignored)).toBe(0)
    expect(everyPath(['ag', 'ag'], ['ag'], ignored)).toBe(0)
    expect(everyPath([], [], ignored)).toBe(1)
    expect(everyPath([], ['ag'], ignored)).toBe(0)
    expect(everyPath([Number.NaN], [Number.NaN], ignored)).toBe(0)
  })
})

describe('occurrence pairing', () => {
  it('pairs repeats in input order and leaves the last one over', () => {
    const evidence = explainer({}).explain(
      ['react', 'react', 'react'],
      ['react', 'react'],
    )
    expect(
      evidence.matches.map((match) => [match.firstIndex, match.secondIndex]),
    ).toEqual([
      [0, 0],
      [1, 1],
    ])
    expect(evidence.unmatchedFirst).toEqual([
      { element: 'react', index: 2, weight: 1, unmatchedMass: 1 },
    ])
    expect(evidence.unmatchedSecond).toEqual([])
  })

  it('orders matches by their first index', () => {
    const evidence = explainer({}).explain(['c', 'b', 'a'], ['a', 'b', 'c'])
    expect(evidence.matches.map((match) => match.firstIndex)).toEqual([0, 1, 2])
    expect(evidence.matches.map((match) => match.secondIndex)).toEqual([2, 1, 0])
  })

  it('leaves the score invariant under permutation while evidence follows order', () => {
    const scorer = explainer({ elementWeights: COMPANY })
    const straight = scorer.explain(['swisscom', 'ag'], ['ag', 'swisscom'])
    const swapped = scorer.explain(['ag', 'swisscom'], ['ag', 'swisscom'])

    expect(straight.similarity).toBe(swapped.similarity)
    expect(straight.totals).toEqual(swapped.totals)
    expect(straight.matches.map((match) => match.first)).toEqual(['swisscom', 'ag'])
    expect(swapped.matches.map((match) => match.first)).toEqual(['ag', 'swisscom'])
  })

  it('reports every match as exact, since matching is', () => {
    const evidence = explainer({ elementWeights: COMPANY }).explain(
      ['swisscom'],
      ['swisscom'],
    )
    expect(evidence.matches).toEqual([
      {
        first: 'swisscom',
        second: 'swisscom',
        firstIndex: 0,
        secondIndex: 0,
        exact: true,
        similarity: 1,
        firstWeight: 5,
        secondWeight: 5,
        sharedMass: 5,
        firstUnmatchedMass: 0,
        secondUnmatchedMass: 0,
      },
    ])
  })
})

describe('raw values out, canonical values for equality', () => {
  it('matches a character against its code point and reports both as given', () => {
    const evidence = explainer({}).explain('ab', [97, 98])
    expect(evidence.similarity).toBe(1)
    expect(evidence.matches.map((match) => [match.first, match.second])).toEqual([
      ['a', 97],
      ['b', 98],
    ])
  })

  it('counts an astral character as one occurrence at one index', () => {
    const evidence = explainer({}).explain('😀a', [0x1f600, 97])
    expect(evidence.similarity).toBe(1)
    expect(evidence.matches.map((match) => [match.first, match.firstIndex])).toEqual([
      ['😀', 0],
      ['a', 1],
    ])
  })

  it('treats a lone surrogate as its own element', () => {
    const evidence = explainer({}).explain('\ud800a', [0xd800, 97])
    expect(evidence.similarity).toBe(1)
    expect(evidence.matches.map((match) => match.firstIndex)).toEqual([0, 1])
  })

  it('reads +0 and -0 as one element', () => {
    const evidence = explainer({}).explain([0], [-0])
    expect(evidence.similarity).toBe(1)
    expect(evidence.matches).toHaveLength(1)
  })

  it('matches objects and symbols by identity, never by structure', () => {
    const shared = { name: 'swisscom' }
    const symbol = Symbol('swisscom')

    expect(explainer({}).explain([shared], [shared]).similarity).toBe(1)
    expect(explainer({}).explain([shared], [{ name: 'swisscom' }]).similarity).toBe(0)
    expect(explainer({}).explain([symbol], [symbol]).similarity).toBe(1)
    expect(explainer({}).explain([symbol], [Symbol('swisscom')]).similarity).toBe(0)
  })

  it('never matches NaN, and leaves it unmatched on its own side', () => {
    const evidence = explainer({}).explain([Number.NaN], [Number.NaN])
    expect(evidence.matches).toEqual([])
    expect(evidence.unmatchedFirst).toEqual([
      { element: Number.NaN, index: 0, weight: 1, unmatchedMass: 1 },
    ])
    expect(evidence.unmatchedSecond).toEqual([
      { element: Number.NaN, index: 0, weight: 1, unmatchedMass: 1 },
    ])
  })
})

describe('elements that contribute nothing', () => {
  it('leaves a zero-weight element out of all three arrays', () => {
    const elementWeights = weightsFor([['ag', 0]])
    const evidence = explainer({ elementWeights }).explain(
      ['swisscom', 'ag'],
      ['swisscom', 'gmbh'],
    )
    expect(evidence.matches.map((match) => match.first)).toEqual(['swisscom'])
    expect(evidence.unmatchedFirst).toEqual([])
    expect(evidence.unmatchedSecond).toEqual([
      { element: 'gmbh', index: 1, weight: 1, unmatchedMass: 1 },
    ])
  })

  it('leaves a zero-weight element out even where both sides hold it', () => {
    const elementWeights = weightsFor([['ag', 0]])
    const evidence = explainer({ elementWeights }).explain(
      ['google', 'ag'],
      ['google', 'ag'],
    )
    expect(evidence.similarity).toBe(1)
    expect(evidence.matches.map((match) => match.first)).toEqual(['google'])
    expect(evidence.unmatchedFirst).toEqual([])
    expect(evidence.unmatchedSecond).toEqual([])
  })
})

describe('the configuration evidence reads', () => {
  it('cannot see a mutation of the caller map the scorer snapshotted', () => {
    const elementWeights = weightsFor([
      ['swisscom', 5],
      ['ag', 0.1],
    ])
    const scorer = explainer({ elementWeights })
    const before = scorer.explain(['swisscom', 'ag'], ['swisscom'])

    elementWeights.set('ag', 99)

    expect(scorer.explain(['swisscom', 'ag'], ['swisscom'])).toEqual(before)
    expect(
      explainer({ elementWeights }).explain(['swisscom', 'ag'], ['swisscom']),
    ).not.toEqual(before)
  })

  // Both spellings of the same thing: a default of 7 with no map, and a map
  // naming every element 7 under that default. Each is one positive weight
  // everywhere, which cancels from the ratio, so the scorer drops the weighted
  // representation — and evidence has to follow it there rather than report
  // masses of 7 for a comparison the scorer ran unweighted.
  for (const [shape, configuration] of [
    ['a default alone', { defaultElementWeight: 7 }],
    [
      'a map naming every element',
      {
        defaultElementWeight: 7,
        elementWeights: weightsFor([
          ['a', 7],
          ['b', 7],
          ['c', 7],
        ]),
      },
    ],
  ] as const) {
    it(`still explains a weighting that compiled away for pricing nothing: ${shape}`, () => {
      const uniform = explainer(configuration)
      const plain = explainer({})
      const evidence = uniform.explain(['a', 'b'], ['b', 'c'])

      expect(evidence.score).toBe(plain.explain(['a', 'b'], ['b', 'c']).score)
      expect(evidence.matches.map((match) => match.firstWeight)).toEqual([1])
      expect(evidence.totals).toEqual(plain.explain(['a', 'b'], ['b', 'c']).totals)
    })
  }

  it('refuses an operand it cannot explain', () => {
    const scorer = explainer({})
    expect(() => callUntyped(scorer.explain, null, ['a'])).toThrow(TypeError)
    expect(() => callUntyped(scorer.explain, ['a'], undefined)).toThrow(TypeError)
    expect(() => callUntyped(scorer.explain, 42, ['a'])).toThrow(TypeError)
  })
})

describe('the capability in the type system', () => {
  it('is granted at gramSize 1 and withheld everywhere else', () => {
    expectTypeOf(
      createScorer(tverskyMetric, { gramSize: 1 }).explain(['a'], ['b']),
    ).toEqualTypeOf<TverskyEvidence>()
    expectTypeOf(
      createScorer(tverskyDistanceMetric, { gramSize: 1 }).explain(['a'], ['b']),
    ).toEqualTypeOf<TverskyEvidence>()

    expectTypeOf(createScorer(tverskyMetric, { gramSize: 2 })).toEqualTypeOf<
      Scorer<'similarity', 'tversky.similarity'>
    >()
    expectTypeOf(createScorer(tverskyMetric)).toEqualTypeOf<
      Scorer<'similarity', 'tversky.similarity'>
    >()
    expectTypeOf(createScorer(levenshteinSimilarity)).toEqualTypeOf<
      Scorer<'similarity', 'levenshtein.similarity'>
    >()
  })

  // The canary for the erasure audit: `ScorerOf` matches on `Metric`'s shape,
  // so a capability parameter it did not account for would silently resolve the
  // whole type to `never` rather than fail anywhere visible.
  it('leaves ScorerOf resolving through a capability-carrying metric', () => {
    expectTypeOf<ScorerOf<typeof tverskyMetric>>().toEqualTypeOf<
      Scorer<'similarity', 'tversky.similarity'>
    >()
    expectTypeOf<ScorerOf<typeof tverskyDistanceMetric>>().toEqualTypeOf<
      Scorer<'distance', 'tversky.distance'>
    >()
  })

  it('needs the literal, which `satisfies` preserves through a variable', () => {
    const widened = { gramSize: 1 }
    const preserved = { gramSize: 1 } satisfies TverskyExplainConfiguration

    expectTypeOf(createScorer(tverskyMetric, widened)).toEqualTypeOf<
      Scorer<'similarity', 'tversky.similarity'>
    >()
    expectTypeOf(
      createScorer(tverskyMetric, preserved).explain(['a'], ['b']),
    ).toEqualTypeOf<TverskyEvidence>()
  })

  // A similarity accepts `missing` and a distance refuses it, so the hoisted
  // form divides the same way the ordinary configuration types do. Named wrong,
  // `missing` is not a member of what is being satisfied and the hoist fails.
  it('names the similarity hoist with its missing-value policy', () => {
    const config = {
      gramSize: 1,
      missing: 'throw',
    } satisfies TverskySimilarityExplainConfiguration

    expectTypeOf(
      createScorer(tverskyMetric, config).explain(['a'], ['b']),
    ).toEqualTypeOf<TverskyEvidence>()
    expect(() =>
      callUntyped(createScorer(tverskyMetric, config).score, null, ['a']),
    ).toThrow(TypeError)
  })
})
