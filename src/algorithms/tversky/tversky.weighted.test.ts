// Weighted exact-element overlap: `elementWeights` at `gramSize: 1`, where a
// gram is one element and each one can price its own contribution.

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'
import { bestMatch, createIndexedMatcher, createMatcher } from '#search/index.js'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { callUntyped } from '../../../testing/untyped.js'
import { tverskyDistance, tverskySimilarity } from './implementation.js'
import {
  distance as tverskyDistanceMetric,
  similarity as tverskyMetric,
} from './index.js'

interface WeightedConfiguration {
  readonly gramSize?: number
  readonly alpha?: number
  readonly beta?: number
  readonly elementWeights?: ReadonlyMap<unknown, number>
  readonly defaultElementWeight?: number
}

function unigram(
  configuration: WeightedConfiguration,
): WeightedConfiguration & Readonly<Record<string, unknown>> {
  return { gramSize: 1, ...configuration }
}

/** The one-shot function, which compiles the weights for this call alone. */
function direct(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  return tverskySimilarity(a, b, unigram(configuration))
}

/** A compiled scorer, which reads the table its canonicalizer snapshotted. */
function configured(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  return createScorer(tverskyMetric, unigram(configuration)).score(a, b)
}

/** The prepared kernel, over a profile built once per choice. */
function prepared(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  return prepareScorerOf(tverskySimilarity)(a, unigram(configuration))(b, null)
}

/** The same kernel through a Matcher, which is what a search reaches. */
function searched(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  const scorer = createScorer(tverskyMetric, unigram(configuration))
  const match = createMatcher([b], { scorer }).best(a)
  return match === undefined ? 0 : match.score
}

/** The inverted index, which reproduces the same scorer over a whole corpus. */
function indexed(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  const scorer = createScorer(tverskyMetric, unigram(configuration))
  const match = createIndexedMatcher([b], { scorer }).best(a)
  return match === undefined ? 0 : match.score
}

/** Every path, which must agree to the bit. */
function everyPath(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: WeightedConfiguration,
): number {
  const score = direct(a, b, configuration)
  expect(configured(a, b, configuration)).toBe(score)
  expect(prepared(a, b, configuration)).toBe(score)
  expect(searched(a, b, configuration)).toBe(score)
  expect(indexed(a, b, configuration)).toBe(score)
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
  ['deepmind', 5],
  ['ag', 0.1],
  ['gmbh', 0.1],
])

describe('weighted exact-element overlap', () => {
  it('prices a generic suffix below a distinctive token', () => {
    const configuration = { elementWeights: COMPANY }
    const withSuffix = everyPath(['swisscom', 'ag'], ['swisscom'], configuration)
    const suffixOnly = everyPath(['swisscom', 'ag'], ['ag'], configuration)
    expect(withSuffix).toBeGreaterThan(0.98)
    expect(suffixOnly).toBeLessThan(0.05)
    // Nothing shared at all still scores 0, weights or not.
    expect(everyPath(['swisscom'], ['google'], configuration)).toBe(0)
  })

  it('weights every occurrence, not every distinct element', () => {
    const configuration = {
      alpha: 1,
      beta: 1,
      elementWeights: weightsFor([['react', 3]]),
    }
    // 6 against 3, shared 3 — one whole weighted repeat is unmatched, which a
    // set-based reading would price at nothing.
    expect(everyPath(['react', 'react'], ['react'], configuration)).toBe(0.5)
    expect(everyPath(['react', 'react'], ['react', 'react'], configuration)).toBe(1)
  })

  it('drops a zero-weight element from the comparison', () => {
    const configuration = {
      elementWeights: weightsFor([
        ['ag', 0],
        ['gmbh', 0],
      ]),
    }
    // `ag` and `gmbh` cost nothing on either side, so only `google` is scored.
    expect(everyPath(['google', 'ag'], ['google', 'gmbh'], configuration)).toBe(1)
  })

  it('scores only the named vocabulary when the default is zero', () => {
    const configuration = {
      defaultElementWeight: 0,
      elementWeights: weightsFor([['typescript', 1]]),
    }
    expect(
      everyPath(['typescript', 'senior'], ['typescript', 'remote'], configuration),
    ).toBe(1)
    expect(everyPath(['senior'], ['remote'], configuration)).toBe(0)
  })

  it('is exactly unweighted Tversky when every weight is one', () => {
    const pairs: readonly (readonly [readonly string[], readonly string[]])[] = [
      [
        ['aa', 'bb'],
        ['aa', 'cc'],
      ],
      [
        ['aa', 'aa', 'bb'],
        ['aa', 'bb', 'bb'],
      ],
      [['aa'], ['aa', 'bb', 'cc']],
      [[], ['aa']],
      [['aa'], []],
    ]
    for (const [a, b] of pairs) {
      for (const [alpha, beta] of [
        [0.5, 0.5],
        [1, 1],
        [1, 0],
        [0.2, 0.7],
      ]) {
        const weighted = direct(a, b, {
          alpha,
          beta,
          elementWeights: weightsFor([
            ['aa', 1],
            ['bb', 1],
            ['cc', 1],
          ]),
        })
        expect(weighted).toBe(tverskySimilarity(a, b, { gramSize: 1, alpha, beta }))
      }
    }
  })
})

describe('weighted arithmetic that a rounded mass would lose', () => {
  it('keeps an unmatched occurrence a weighted mass would have absorbed', () => {
    // massA folds 1e16 + 1 back to 1e16, so `massA - shared` would report a
    // perfect match while one whole token is missing.
    expect(
      everyPath(['x', 'y'], ['x'], {
        alpha: 1e16,
        beta: 0,
        elementWeights: weightsFor([
          ['x', 1e16],
          ['y', 1],
        ]),
      }),
    ).toBe(0.5)
  })

  it('scores an identical pair 1 when the largest weight underflows the shared mass', () => {
    // `shared / max(1, alpha)` is 0 here, and the ordinary ratio would be 0 / 0.
    expect(
      everyPath(['x'], ['x'], {
        alpha: Number.MAX_VALUE,
        beta: 0,
        elementWeights: weightsFor([['x', Number.MIN_VALUE]]),
      }),
    ).toBe(1)
  })

  it('is invariant to scaling every weight by one power of two', () => {
    const scale = 2 ** 40
    const plain = weightsFor([
      ['aa', 1],
      ['bb', 3],
      ['cc', 0.25],
    ])
    const scaled = weightsFor(
      [...plain].map(([element, weight]) => [element, weight * scale]),
    )
    for (const [alpha, beta] of [
      [0.5, 0.5],
      [1, 0],
      [2, 10],
    ]) {
      expect(
        direct(['aa', 'bb', 'cc'], ['aa', 'cc', 'cc'], {
          alpha,
          beta,
          elementWeights: scaled,
          defaultElementWeight: scale,
        }),
      ).toBe(
        direct(['aa', 'bb', 'cc'], ['aa', 'cc', 'cc'], {
          alpha,
          beta,
          elementWeights: plain,
        }),
      )
    }
  })
})

describe('weighted metamorphic invariants', () => {
  const vocabulary = ['aa', 'bb', 'cc', 'dd'] as const
  const hostileWeights = [1e16, 1, 0.1, Number.EPSILON, 1e-300] as const

  const tokens = fc.array(fc.constantFrom(...vocabulary), { maxLength: 6 })
  const weightTuple = fc.tuple(
    fc.constantFrom(...hostileWeights),
    fc.constantFrom(...hostileWeights),
    fc.constantFrom(...hostileWeights),
    fc.constantFrom(...hostileWeights),
  )
  const weightPair = fc
    .tuple(
      fc.constantFrom(0, 0.1, 0.5, 1, 2, 1e16),
      fc.constantFrom(0, 0.1, 0.5, 1, 2, 1e16),
    )
    .filter(([alpha, beta]) => alpha !== 0 || beta !== 0)

  function mapOf(values: readonly number[]): Map<unknown, number> {
    return weightsFor(vocabulary.map((element, index) => [element, values[index]]))
  }

  function shuffled(values: readonly string[], seed: number): string[] {
    // MINSTD, so `seed` actually chooses a permutation: a state below 2³¹ times
    // 48271 stays inside exact integer range, and never reaches zero.
    let state = seed + 1
    const copy = [...values]
    for (let at = copy.length - 1; at > 0; at--) {
      state = (state * 48_271) % 2_147_483_647
      const other = state % (at + 1)
      const held = copy[at]
      copy[at] = copy[other]
      copy[other] = held
    }
    return copy
  }

  it('reads a permutation of either side the same', () => {
    fc.assert(
      fc.property(
        tokens,
        tokens,
        weightTuple,
        weightPair,
        fc.nat(97),
        (a, b, values, [alpha, beta], seed) => {
          const configuration = { alpha, beta, elementWeights: mapOf(values) }
          expect(direct(shuffled(a, seed), shuffled(b, seed + 1), configuration)).toBe(
            direct(a, b, configuration),
          )
        },
      ),
    )
  })

  it('is symmetric exactly when the two weights are equal', () => {
    fc.assert(
      fc.property(
        tokens,
        tokens,
        weightTuple,
        fc.constantFrom(0.5, 1, 2, 1e16),
        (a, b, values, weight) => {
          const configuration = {
            alpha: weight,
            beta: weight,
            elementWeights: mapOf(values),
          }
          expect(direct(a, b, configuration)).toBe(direct(b, a, configuration))
        },
      ),
    )
  })

  it('swaps the arguments exactly as it swaps the weights', () => {
    fc.assert(
      fc.property(
        tokens,
        tokens,
        weightTuple,
        weightPair,
        (a, b, values, [alpha, beta]) => {
          const elementWeights = mapOf(values)
          expect(direct(a, b, { alpha, beta, elementWeights })).toBe(
            direct(b, a, { alpha: beta, beta: alpha, elementWeights }),
          )
        },
      ),
    )
  })

  it('scores a permutation of one sequence against itself 1, and stays in 0..1', () => {
    fc.assert(
      fc.property(
        tokens,
        weightTuple,
        weightPair,
        fc.nat(97),
        (a, values, [alpha, beta], seed) => {
          const configuration = { alpha, beta, elementWeights: mapOf(values) }
          const self = direct(a, shuffled(a, seed), configuration)
          expect(self).toBe(1)
          expect(self).toBeGreaterThanOrEqual(0)
          expect(self).toBeLessThanOrEqual(1)
        },
      ),
    )
  })
})

describe('the integer-weight expansion oracle', () => {
  const vocabulary = ['aa', 'bb', 'cc'] as const

  function expand(
    values: readonly string[],
    weights: ReadonlyMap<unknown, number>,
  ): string[] {
    const expanded: string[] = []
    for (const element of values) {
      const weight = weights.get(element)
      for (let at = 0; at < (weight === undefined ? 1 : weight); at++) {
        expanded.push(element)
      }
    }
    return expanded
  }

  function massOf(
    values: readonly string[],
    weights: ReadonlyMap<unknown, number>,
  ): number {
    let mass = 0
    for (const element of values) {
      const weight = weights.get(element)
      mass += weight === undefined ? 1 : weight
    }
    return mass
  }

  it('equals ordinary Tversky over the expanded sequences', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...vocabulary), { maxLength: 6 }),
        fc.array(fc.constantFrom(...vocabulary), { maxLength: 6 }),
        fc.tuple(fc.nat(4), fc.nat(4), fc.nat(4)),
        fc.constantFrom(1, 2, 3),
        fc.tuple(fc.constantFrom(0, 0.1, 0.5, 1, 2), fc.constantFrom(0, 0.1, 0.5, 1, 2)),
        (a, b, values, defaultElementWeight, [alpha, beta]) => {
          if (alpha === 0 && beta === 0) return
          const elementWeights = weightsFor(
            vocabulary.map((element, index) => [element, values[index]]),
          )
          const configuration = { alpha, beta, elementWeights, defaultElementWeight }
          // Expansion says nothing about the both-zero-mass domain: it maps two
          // different ignored sequences to two empty ones, which ordinary
          // Tversky calls equal and the weighted rule deliberately does not.
          // One side with mass is still a valid oracle — empty against
          // non-empty scores 0 either way.
          if (massOf(a, elementWeights) === 0 && massOf(b, elementWeights) === 0) return
          // Nor about a uniform positive weighting, which is deliberately scored
          // as ordinary unweighted Tversky rather than as `w` copies of every
          // element: those are the same real number and may round one ulp apart.
          // What that case owes instead is `uniform === unweighted`, asserted in
          // `weighted configuration`.
          if (values.every((weight) => weight === defaultElementWeight)) return
          expect(direct(a, b, configuration)).toBe(
            tverskySimilarity(expand(a, elementWeights), expand(b, elementWeights), {
              gramSize: 1,
              alpha,
              beta,
            }),
          )
        },
      ),
    )
  })
})

describe('weighted zero mass', () => {
  const ignored = { defaultElementWeight: 0 }

  it('is 1 for equal multisets in any order, and 0 for anything else', () => {
    expect(everyPath(['ag'], ['ag'], ignored)).toBe(1)
    expect(everyPath(['ag', 'gmbh'], ['gmbh', 'ag'], ignored)).toBe(1)
    expect(everyPath(['ag'], ['gmbh'], ignored)).toBe(0)
    expect(everyPath(['ag'], ['ag', 'gmbh'], ignored)).toBe(0)
    expect(everyPath(['ag', 'ag'], ['ag'], ignored)).toBe(0)
    expect(everyPath([], [], ignored)).toBe(1)
    expect(everyPath([], ['ag'], ignored)).toBe(0)
  })

  it('is 0 whenever one side carries mass and the other does not', () => {
    const configuration = { elementWeights: weightsFor([['ag', 0]]) }
    expect(everyPath(['ag'], ['google'], configuration)).toBe(0)
    expect(everyPath(['google'], ['ag'], configuration)).toBe(0)
  })

  it('never calls two sequences holding NaN equal', () => {
    expect(everyPath([Number.NaN], [Number.NaN], ignored)).toBe(0)
  })

  it('does not let ignored suffixes alone make a perfect entity match', () => {
    const configuration = {
      elementWeights: weightsFor([
        ['ag', 0],
        ['gmbh', 0],
      ]),
    }
    expect(everyPath(['ag'], ['gmbh'], configuration)).toBe(0)
  })
})

describe('weighted element semantics', () => {
  it('reads a string and its code points as the same sequence', () => {
    const elementWeights = weightsFor([
      ['a', 4],
      ['b', 1],
    ])
    expect(direct('abc', 'abd', { elementWeights })).toBe(
      direct([97, 98, 99], [97, 98, 100], { elementWeights }),
    )
    // And naming a weight by code point is naming the character.
    expect(
      direct('ab', 'a', {
        elementWeights: weightsFor([
          [97, 4],
          [98, 1],
        ]),
      }),
    ).toBe(direct('ab', 'a', { elementWeights }))
  })

  it('holds objects by identity and never by structure', () => {
    const first = { name: 'react' }
    const second = { name: 'react' }
    const elementWeights = weightsFor([[first, 4]])
    expect(everyPath([first], [first], { elementWeights })).toBe(1)
    expect(everyPath([first], [second], { elementWeights })).toBe(0)
  })

  it('prices an unmatchable element without ever sharing it', () => {
    const elementWeights = weightsFor([['aa', 1]])
    // NaN counts toward its own side and matches nothing, so a query holding
    // one can never be fully contained.
    expect(
      direct(['aa', Number.NaN], ['aa'], { alpha: 1, beta: 0, elementWeights }),
    ).toBe(0.5)
    // Weighing it zero is the way to ignore it.
    expect(
      direct(['aa', Number.NaN], ['aa'], {
        alpha: 1,
        beta: 0,
        elementWeights: weightsFor([
          ['aa', 1],
          [Number.NaN, 0],
        ]),
      }),
    ).toBe(1)
  })

  it('treats +0 and -0 as one element, as every other structure here does', () => {
    expect(direct([0], [-0], { elementWeights: weightsFor([[0, 3]]) })).toBe(1)
    expect(direct([0], [0], { elementWeights: weightsFor([[-0, 3]]) })).toBe(1)
  })

  it('takes symbols and mixed primitives as elements', () => {
    const token = Symbol('react')
    const elementWeights = weightsFor([
      [token, 5],
      [true, 0.5],
      ['aa', 2],
    ])
    expect(
      everyPath([token, true, 'aa'], [token, 'aa'], { elementWeights }),
    ).toBeGreaterThan(0.9)
    expect(everyPath([token], [Symbol('react')], { elementWeights })).toBe(0)
  })

  it('counts an unknown query element against the query alone', () => {
    const elementWeights = weightsFor([['aa', 1]])
    expect(direct(['aa', 'zz'], ['aa'], { alpha: 1, beta: 0, elementWeights })).toBe(0.5)
  })
})

describe('weighted configuration', () => {
  it('is only defined at gramSize 1', () => {
    for (const gramSize of [2, 3]) {
      expect(() =>
        tverskySimilarity(['aa'], ['aa'], {
          gramSize,
          elementWeights: weightsFor([['aa', 2]]),
        }),
      ).toThrow('only defined at gramSize 1')
      expect(() =>
        createScorer(tverskyMetric, { gramSize, defaultElementWeight: 2 }),
      ).toThrow(RangeError)
    }
    // The default gram size is 2, so weights without one are refused too.
    expect(() =>
      createScorer(tverskyMetric, { elementWeights: weightsFor([['aa', 2]]) }),
    ).toThrow('only defined at gramSize 1')
  })

  it('refuses a weights argument that is not map-like', () => {
    for (const value of [1, 'ab', null, [5, 10], new Set(['aa'])]) {
      expect(() =>
        callUntyped(createScorer, tverskyMetric, {
          gramSize: 1,
          elementWeights: value,
        }),
      ).toThrow('elementWeights must be a map')
    }
  })

  it('refuses weights that are not finite non-negative numbers', () => {
    for (const weight of [Number.NaN, Infinity, -1]) {
      expect(() =>
        createScorer(tverskyMetric, {
          gramSize: 1,
          elementWeights: weightsFor([['aa', weight]]),
        }),
      ).toThrow(RangeError)
    }
    expect(() =>
      callUntyped(createScorer, tverskyMetric, {
        gramSize: 1,
        defaultElementWeight: null,
      }),
    ).toThrow('defaultElementWeight must be a number')
  })

  it('refuses one element named twice with different weights', () => {
    expect(() =>
      createScorer(tverskyMetric, {
        gramSize: 1,
        elementWeights: weightsFor([
          ['a', 2],
          [97, 4],
        ]),
      }),
    ).toThrow('one element two weights')
  })

  it('refuses a weight span too wide to represent', () => {
    expect(() =>
      createScorer(tverskyMetric, {
        gramSize: 1,
        elementWeights: weightsFor([
          ['huge', Number.MAX_VALUE],
          ['tiny', Number.MIN_VALUE],
        ]),
      }),
    ).toThrow('too wide to represent')
  })

  it('snapshots the map, so a later mutation changes nothing', () => {
    const weights = weightsFor([['ag', 0.1]])
    const scorer = createScorer(tverskyMetric, {
      gramSize: 1,
      elementWeights: weights,
      alpha: 1,
      beta: 0,
    })
    const matcher = createMatcher([['swisscom', 'ag']], { scorer })
    const before = scorer.score(['swisscom', 'ag'], ['swisscom'])
    const matchedBefore = matcher.best(['swisscom'])?.score
    weights.set('ag', 100)
    expect(scorer.score(['swisscom', 'ag'], ['swisscom'])).toBe(before)
    expect(matcher.best(['swisscom'])?.score).toBe(matchedBefore)
    // A scorer created after the mutation reads the new weight.
    expect(
      createScorer(tverskyMetric, {
        gramSize: 1,
        elementWeights: weights,
        alpha: 1,
        beta: 0,
      }).score(['swisscom', 'ag'], ['swisscom']),
    ).not.toBe(before)
  })

  it('never shares prepared choices with another weight configuration', () => {
    const weighted = createScorer(tverskyMetric, {
      gramSize: 1,
      elementWeights: weightsFor([['ag', 0.1]]),
    })
    const other = createScorer(tverskyMetric, {
      gramSize: 1,
      elementWeights: weightsFor([['ag', 0.2]]),
    })
    const plain = createScorer(tverskyMetric, { gramSize: 1 })
    const rows = [{ prepared: weighted.prepareChoice(['swisscom', 'ag']) }]
    expect(
      bestMatch(['swisscom'], rows, {
        scorer: weighted,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBeGreaterThan(0)
    for (const scorer of [other, plain]) {
      expect(() =>
        bestMatch(['swisscom'], rows, { scorer, getPrepared: (row) => row.prepared }),
      ).toThrow('prepared choice is incompatible with this scorer')
    }
  })

  it('keeps refusing an unknown configuration key', () => {
    expect(() =>
      callUntyped(createScorer, tverskyMetric, { gramSize: 1, elementWeight: 2 }),
    ).toThrow("unknown metric configuration key 'elementWeight'")
  })

  it('reports symmetry from the weight pair, as the unweighted metric does', () => {
    // Element weights price both sides alike, so they cannot make an asymmetric
    // scorer symmetric or the reverse: `alpha === beta` is still the whole rule.
    const elementWeights = weightsFor([['ag', 0.1]])
    expect(createScorer(tverskyMetric, { gramSize: 1, elementWeights }).symmetric).toBe(
      true,
    )
    expect(
      createScorer(tverskyMetric, { gramSize: 1, alpha: 1, beta: 0, elementWeights })
        .symmetric,
    ).toBe(false)
  })

  it('drops a weighting where every element weighs the same', () => {
    // One constant factor over all three components cancels from the ratio, so
    // these are ordinary unigram Tversky and are scored by the unweighted
    // engines: the same number through every path, from the same profiles.
    const plain = createScorer(tverskyMetric, { gramSize: 1, alpha: 1, beta: 0.1 })
    const uniform: readonly WeightedConfiguration[] = [
      { defaultElementWeight: 1 },
      { elementWeights: weightsFor([['swisscom', 1]]) },
      {
        defaultElementWeight: 7,
        elementWeights: weightsFor([
          ['swisscom', 7],
          ['ag', 7],
        ]),
      },
    ]
    const rows = [['swisscom', 'ag']]
    for (const configuration of uniform) {
      const scorer = createScorer(tverskyMetric, {
        gramSize: 1,
        alpha: 1,
        beta: 0.1,
        ...configuration,
      })
      expect(scorer.score(['swisscom'], rows[0])).toBe(plain.score(['swisscom'], rows[0]))
      expect(
        direct(['swisscom'], rows[0], { alpha: 1, beta: 0.1, ...configuration }),
      ).toBe(plain.score(['swisscom'], rows[0]))
      // And its own prepared and indexed paths agree with it, which is what says
      // the unweighted preparation is the one that ran.
      expect(createMatcher(rows, { scorer }).best(['swisscom'])?.score).toBe(
        plain.score(['swisscom'], rows[0]),
      )
      expect(createIndexedMatcher(rows, { scorer }).best(['swisscom'])?.score).toBe(
        plain.score(['swisscom'], rows[0]),
      )
      // Its prepared representation is the ordinary unigram profile rather than
      // a weighted one, which is why its own prepared search agrees. Scorer
      // identity stays configuration-scoped as it is everywhere else: a
      // separately created scorer owns the handles it prepared, so the plain one
      // still refuses these.
      const handles = [{ prepared: scorer.prepareChoice(rows[0]) }]
      const read = (used: typeof scorer) =>
        bestMatch(['swisscom'], handles, {
          scorer: used,
          getPrepared: (row) => row.prepared,
        })
      expect(read(scorer)?.score).toBe(plain.score(['swisscom'], rows[0]))
      expect(() => read(plain)).toThrow('prepared choice is incompatible')
    }
    // Still refused at the wrong gram size, and a zero default is not uniform:
    // ignored elements have their own semantics.
    expect(() =>
      createScorer(tverskyMetric, { gramSize: 2, defaultElementWeight: 1 }),
    ).toThrow('only defined at gramSize 1')
    expect(everyPath(['ag'], ['gmbh'], { defaultElementWeight: 0 })).toBe(0)
  })
})

describe('weighted distance', () => {
  it('is one minus the weighted similarity, through every path', () => {
    const configuration = { gramSize: 1, elementWeights: COMPANY, alpha: 1, beta: 0 }
    const a = ['swisscom', 'ag']
    const b = ['swisscom']
    const similarity = tverskySimilarity(a, b, configuration)
    expect(tverskyDistance(a, b, configuration)).toBe(1 - similarity)
    expect(createScorer(tverskyDistanceMetric, configuration).score(a, b)).toBe(
      1 - similarity,
    )
    expect(prepareScorerOf(tverskyDistance)(a, configuration)(b, null)).toBe(
      1 - similarity,
    )
  })

  it('prunes a prepared candidate the threshold rules out, in both directions', () => {
    const choices = [
      ['swisscom', 'ag'],
      ['google', 'gmbh'],
    ]
    // beta above zero, so a covered query still pays for the extra token and
    // the best score stays under 1 — containment alone would reach it.
    const configuration = { gramSize: 1, elementWeights: COMPANY, alpha: 1, beta: 0.1 }
    const scorer = createScorer(tverskyMetric, configuration)
    const score = scorer.score(['swisscom'], choices[0])
    expect(score).toBeLessThan(1)
    expect(
      createMatcher(choices, { scorer }).search(['swisscom'], { threshold: score }),
    ).toHaveLength(1)
    // Above the best score, the kernel answers a miss rather than a number.
    expect(
      createMatcher(choices, { scorer }).search(['swisscom'], { threshold: 1 }),
    ).toEqual([])
    const distanceScorer = createScorer(tverskyDistanceMetric, configuration)
    expect(
      createMatcher(choices, { scorer: distanceScorer }).search(['swisscom'], {
        threshold: 1 - score,
      }),
    ).toHaveLength(1)
    expect(
      createMatcher(choices, { scorer: distanceScorer }).search(['swisscom'], {
        threshold: 0,
      }),
    ).toEqual([])
  })

  it('answers a threshold on either direction', () => {
    const configuration = { gramSize: 1, elementWeights: COMPANY }
    const scorer = createScorer(tverskyMetric, configuration)
    const score = scorer.score(['swisscom', 'ag'], ['swisscom'])
    expect(scorer.score(['swisscom', 'ag'], ['swisscom'], { threshold: score })).toBe(
      score,
    )
    expect(
      scorer.score(['swisscom', 'ag'], ['swisscom'], { threshold: 1 }),
    ).toBeUndefined()
  })
})
