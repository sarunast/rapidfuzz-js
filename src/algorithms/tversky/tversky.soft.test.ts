// Fuzzy element matching at `gramSize: 1`: exact overlap first, then a
// one-to-one matching of what it left over, and the contract that a threshold
// nothing reaches is the exact scorer to the bit.

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { scoreMatrix } from '#batch/scoreMatrix.js'
import { createScorer, type Scorer } from '#core/scoring/scorer.js'
import { bestMatch, createIndexedMatcher, createMatcher } from '#search/index.js'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { normalizedSimilarity as indelSimilarity } from '../indel/index.js'
import { tverskyDistance, tverskySimilarity } from './implementation.js'
import {
  distance as tverskyDistanceMetric,
  similarity as tverskyMetric,
} from './index.js'

const indel = createScorer(indelSimilarity)

interface SoftConfiguration {
  readonly gramSize?: number
  readonly alpha?: number
  readonly beta?: number
  readonly elementWeights?: ReadonlyMap<unknown, number>
  readonly defaultElementWeight?: number
  readonly elementSimilarity?: { scorer: Scorer<'similarity'>; threshold: number }
  readonly scoreCutoff?: number
}

function unigram(
  configuration: SoftConfiguration,
): SoftConfiguration & Readonly<Record<string, unknown>> {
  return { gramSize: 1, ...configuration }
}

function direct(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: SoftConfiguration,
): number {
  return tverskySimilarity(a, b, unigram(configuration))
}

function configured(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: SoftConfiguration,
): number {
  return createScorer(tverskyMetric, unigram(configuration)).score(a, b)
}

function prepared(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: SoftConfiguration,
): number {
  return prepareScorerOf(tverskySimilarity)(a, unigram(configuration))(b, null)
}

function matched(
  a: readonly unknown[] | string,
  b: readonly unknown[] | string,
  configuration: SoftConfiguration,
): number {
  const scorer = createScorer(tverskyMetric, unigram(configuration))
  return createMatcher([b], { scorer }).best(a)?.score ?? Number.NaN
}

/** Every path a score can arrive by, so a divergence names itself. */
const PATHS = [
  ['one-shot', direct],
  ['configured', configured],
  ['prepared', prepared],
  ['matcher', matched],
] as const

/** An element scorer that only ever pairs the two tokens it is told to. */
function pairsOnly(pairs: ReadonlyArray<readonly [string, string, number]>) {
  return createScorer(
    (a, b) => {
      for (const [one, other, score] of pairs) {
        if ((a === one && b === other) || (a === other && b === one)) return score
      }
      return a === b ? 1 : 0
    },
    { direction: 'similarity', bounds: [0, 1], symmetric: true },
  )
}

const SOFT = { scorer: indel, threshold: 0.8 }

describe('an unreachable threshold is the exact scorer', () => {
  // The acceptance test. `1` admits only elements the inner scorer calls
  // identical, and no leftover pair ever is, so no edge can survive.
  const UNREACHABLE = { scorer: indel, threshold: 1 }

  const CASES: ReadonlyArray<readonly [string, readonly unknown[], readonly unknown[]]> =
    [
      ['disjoint tokens', ['google', 'deepmind'], ['swisscom', 'ag']],
      ['partly shared', ['swisscom', 'ag'], ['swisscom', 'gmbh']],
      ['repeated tokens', ['react', 'react', 'vue'], ['react', 'angular']],
      ['one side empty', [], ['swisscom']],
      ['both empty', [], []],
      ['unmatchable', ['react', Number.NaN], ['react']],
      ['objects by identity', [{ id: 1 }, 'react'], ['react']],
      ['single characters', ['a', 'b'], ['a', 'c']],
    ]

  describe.each(PATHS)('%s', (_name, score) => {
    it.each(CASES)('scores %s identically', (_label, a, b) => {
      expect(score(a, b, { elementSimilarity: UNREACHABLE })).toBe(score(a, b, {}))
    })

    it.each(CASES)('scores %s identically when weighted', (_label, a, b) => {
      const elementWeights = new Map<unknown, number>([
        ['swisscom', 5],
        ['ag', 0.1],
        ['react', 2],
      ])
      expect(score(a, b, { elementSimilarity: UNREACHABLE, elementWeights })).toBe(
        score(a, b, { elementWeights }),
      )
    })
  })

  it('agrees on a distance scorer too', () => {
    const a = ['swisscom', 'ag']
    const b = ['swisscom', 'gmbh']
    expect(tverskyDistance(a, b, unigram({ elementSimilarity: UNREACHABLE }))).toBe(
      tverskyDistance(a, b, unigram({})),
    )
  })
})

describe('fuzzy element matching', () => {
  it('closes the gap a typo opens', () => {
    const exact = direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {})
    const soft = direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
      elementSimilarity: SOFT,
    })
    // Only `ag` is shared, so at the Dice defaults the pair scores a flat half.
    expect(exact).toBe(0.5)
    expect(soft).toBeGreaterThan(0.9)
  })

  it.each(PATHS)('reaches the same score through the %s path', (_name, score) => {
    expect(
      score(['swisscom', 'ag'], ['swisscomm', 'ag'], { elementSimilarity: SOFT }),
    ).toBe(direct(['swisscom', 'ag'], ['swisscomm', 'ag'], { elementSimilarity: SOFT }))
  })

  it('never scores below the exact scorer', () => {
    const tokens = fc.array(
      fc.constantFrom('swisscom', 'swisscomm', 'ag', 'gmbh', 'google', 'googel'),
      { maxLength: 5 },
    )
    fc.assert(
      fc.property(tokens, tokens, (a, b) => {
        const soft = direct(a, b, { elementSimilarity: SOFT })
        expect(soft).toBeGreaterThanOrEqual(direct(a, b, {}) - 1e-12)
      }),
      { numRuns: 500, seed: 0x5eed },
    )
  })

  it('admits a pair exactly at the threshold and refuses one a hair below', () => {
    const similarity = 0.5
    const scorer = pairsOnly([['alpha', 'beta', similarity]])
    const admitted = direct(['alpha'], ['beta'], {
      elementSimilarity: { scorer, threshold: similarity },
    })
    const refused = direct(['alpha'], ['beta'], {
      elementSimilarity: { scorer, threshold: similarity + Number.EPSILON / 2 },
    })
    expect(admitted).toBeGreaterThan(0)
    expect(refused).toBe(0)
  })

  // Weights 4 and 1 with similarity 0.5 share `min(4, 1) × 0.5`, leaving the
  // heavy side owing 3.5 and the light one 0.5 — whichever side is heavy.
  it.each([
    ['the first side is heavier', 4, 1],
    ['the second side is heavier', 1, 4],
  ])('shares only the smaller weight when %s', (_label, first, second) => {
    const scorer = pairsOnly([['alpha', 'beta', 0.5]])
    const score = direct(['alpha'], ['beta'], {
      alpha: 1,
      beta: 1,
      elementWeights: new Map([
        ['alpha', first],
        ['beta', second],
      ]),
      elementSimilarity: { scorer, threshold: 0.5 },
    })
    expect(score).toBe(0.5 / (0.5 + (3.5 + 0.5)))
  })
})

// The solver has an oracle of its own, but the solver is one stage of four:
// exact reservation, edge building, the matching, and the fold into a Tversky
// ratio. Only a brute force over the whole pipeline can catch a stage that
// disagrees with the ones around it.
describe('against an exhaustive whole-pipeline oracle', () => {
  const VOCABULARY = ['north', 'nordh', 'south', 'sowth'] as const

  // Dyadic, so every partial sum the oracle forms is exact and a disagreement
  // is a real one rather than a fold order.
  const ALIKE: ReadonlyArray<readonly [string, string, number]> = [
    ['north', 'nordh', 0.75],
    ['south', 'sowth', 0.5],
    ['north', 'south', 0.25],
    ['nordh', 'sowth', 0.5],
  ]
  const ORACLE_THRESHOLD = 0.5
  const oracleScorer = pairsOnly(ALIKE)

  function similarityOf(a: string, b: string): number {
    if (a === b) return 1
    for (const [one, other, score] of ALIKE) {
      if ((a === one && b === other) || (a === other && b === one)) return score
    }
    return 0
  }

  function sequences(maxLength: number): string[][] {
    const out: string[][] = []
    const walk = (prefix: string[]): void => {
      if (prefix.length > 0) out.push(prefix)
      if (prefix.length === maxLength) return
      for (const token of VOCABULARY) walk([...prefix, token])
    }
    walk([])
    return out
  }

  function tally(tokens: readonly string[]): Map<string, number> {
    const counts = new Map<string, number>()
    for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1)
    return counts
  }

  /**
   * Every matching of the leftovers, one occurrence at a time. The objective is
   * the shared mass alone: both penalties follow from it, since a matched
   * occurrence splits its own weight between shared and residual.
   */
  function bestMatching(
    leftFirst: readonly string[],
    leftSecond: readonly string[],
    weightOf: (token: string) => number,
  ): { shared: number; firstOnly: number; secondOnly: number } {
    const taken = leftSecond.map(() => false)
    let best = { shared: -1, firstOnly: 0, secondOnly: 0 }
    const walk = (at: number, shared: number, firstOnly: number, secondOnly: number) => {
      if (at === leftFirst.length) {
        let tail = secondOnly
        for (let j = 0; j < leftSecond.length; j++) {
          if (!taken[j]) tail += weightOf(leftSecond[j])
        }
        if (shared > best.shared) best = { shared, firstOnly, secondOnly: tail }
        return
      }
      const token = leftFirst[at]
      const weight = weightOf(token)
      walk(at + 1, shared, firstOnly + weight, secondOnly)
      for (let j = 0; j < leftSecond.length; j++) {
        if (taken[j]) continue
        const other = leftSecond[j]
        const similarity = similarityOf(token, other)
        if (!(similarity >= ORACLE_THRESHOLD)) continue
        const otherWeight = weightOf(other)
        const profit = Math.min(weight, otherWeight) * similarity
        if (profit === 0) continue
        taken[j] = true
        walk(
          at + 1,
          shared + profit,
          firstOnly + (weight - profit),
          secondOnly + (otherWeight - profit),
        )
        taken[j] = false
      }
    }
    walk(0, 0, 0, 0)
    return best
  }

  function oracle(
    first: readonly string[],
    second: readonly string[],
    weightOf: (token: string) => number,
    alpha: number,
    beta: number,
  ): number {
    const countsFirst = tally(first)
    const countsSecond = tally(second)
    let shared = 0
    const leftFirst: string[] = []
    const leftSecond: string[] = []
    for (const [token, count] of countsFirst) {
      const reserved = Math.min(count, countsSecond.get(token) ?? 0)
      shared += reserved * weightOf(token)
      for (let at = reserved; at < count; at++) leftFirst.push(token)
    }
    for (const [token, count] of countsSecond) {
      const reserved = Math.min(count, countsFirst.get(token) ?? 0)
      for (let at = reserved; at < count; at++) leftSecond.push(token)
    }
    const fuzzy = bestMatching(leftFirst, leftSecond, weightOf)
    const total = shared + fuzzy.shared
    return total / (total + alpha * fuzzy.firstOnly + beta * fuzzy.secondOnly)
  }

  const WEIGHTS = new Map<unknown, number>([
    ['north', 4],
    ['nordh', 2],
    ['south', 1],
    ['sowth', 1],
  ])

  const CONFIGURATIONS: ReadonlyArray<
    readonly [string, SoftConfiguration, (token: string) => number]
  > = [
    ['unweighted at the defaults', { alpha: 1, beta: 1 }, () => 1],
    [
      'weighted at the defaults',
      { alpha: 1, beta: 1, elementWeights: WEIGHTS },
      (token) => WEIGHTS.get(token) ?? 1,
    ],
    ['unweighted and asymmetric', { alpha: 2, beta: 0.5 }, () => 1],
  ]

  it.each(CONFIGURATIONS)('reproduces the optimum %s', (_name, base, weightOf) => {
    const all = sequences(3)
    const configuration = {
      ...base,
      elementSimilarity: { scorer: oracleScorer, threshold: ORACLE_THRESHOLD },
    }
    const scorer = createScorer(tverskyMetric, unigram(configuration))
    const exactScorer = createScorer(tverskyMetric, unigram(base))
    let lifted = 0
    for (const first of all) {
      for (const second of all) {
        const expected = oracle(first, second, weightOf, base.alpha ?? 1, base.beta ?? 1)
        const actual = scorer.score(first, second)
        const exact = exactScorer.score(first, second)
        expect(actual).toBeCloseTo(expected, 12)
        expect(actual).toBeGreaterThanOrEqual(exact - 1e-12)
        if (actual > exact + 1e-12) lifted++
      }
    }
    // An oracle both sides pass by never matching anything proves nothing, so
    // the grid has to be one where fuzzy matching really moves the score.
    expect(lifted).toBeGreaterThan(all.length ** 2 / 2)
  })
})

describe('exact matching is reserved first', () => {
  // Unconstrained, `left↔shared` and `shared↔right` would share two whole
  // tokens; reserving the exact pair leaves one edge that cannot be used.
  const scorer = pairsOnly([
    ['left-token', 'shared-token', 1],
    ['shared-token', 'right-token', 1],
    ['left-token', 'right-token', 0],
  ])
  const configuration = {
    alpha: 1,
    beta: 1,
    elementSimilarity: { scorer, threshold: 0.5 },
  }

  it('pairs the equal tokens even where another pairing would score higher', () => {
    const score = direct(
      ['shared-token', 'left-token'],
      ['shared-token', 'right-token'],
      configuration,
    )
    // Reserved: 1 shared of 2 each side → 1 / (1 + 1 + 1).
    expect(score).toBe(1 / 3)
    // The unconstrained optimum would have shared both tokens outright.
    expect(score).not.toBe(1)
  })

  it('leaves a token free for fuzzy matching once nothing equal remains', () => {
    const score = direct(['left-token'], ['shared-token'], configuration)
    expect(score).toBe(1)
  })
})

describe('multiplicity', () => {
  it('collapses repeats into one comparison and still spreads the units', () => {
    // Dyadic similarities, so every partial sum below is exact and the
    // expectation cannot disagree with the fold in the last bit.
    const scorer = pairsOnly([
      ['react', 'reakt', 1],
      ['react', 'reactt', 0.75],
    ])
    const configuration = {
      alpha: 1,
      beta: 1,
      elementSimilarity: { scorer, threshold: 0.5 },
    }
    // Two `react` occurrences reach two different counterparts, sharing
    // 1 + 0.75; each side is then left owing the 0.25 the weaker pair did not
    // cover. The two penalties are summed before the numerator joins, exactly
    // as the scorer does.
    expect(direct(['react', 'react'], ['reakt', 'reactt'], configuration)).toBe(
      1.75 / (1.75 + (0.25 + 0.25)),
    )
  })

  it('lets only one competitor claim a single occurrence', () => {
    const scorer = pairsOnly([
      ['swisscom', 'swisscomm', 1],
      ['swisscoma', 'swisscomm', 1],
    ])
    const score = direct(['swisscom', 'swisscoma'], ['swisscomm'], {
      alpha: 1,
      beta: 1,
      elementSimilarity: { scorer, threshold: 0.5 },
    })
    // One of the two is matched whole; the other is entirely unmatched.
    expect(score).toBe(1 / (1 + 1))
  })
})

describe('what an element scorer never sees', () => {
  const scorer = pairsOnly([
    ['a', 'ab', 1],
    ['😀', '😀a', 1],
  ])
  const configuration = { elementSimilarity: { scorer, threshold: 0.5 } }

  it.each([
    ['a single-character token', ['a'], ['ab']],
    ['an astral single-code-point token', ['😀'], ['😀a']],
    ['a code point given as a number', [97], ['ab']],
    ['an array-valued token', [['a', 'b']], ['ab']],
  ])('leaves %s to exact matching', (_label, a, b) => {
    expect(direct(a, b, configuration)).toBe(direct(a, b, {}))
  })

  it('leaves a plain string exactly where it was', () => {
    expect(direct('abc', 'abd', configuration)).toBe(direct('abc', 'abd', {}))
  })

  it('still prices unmatchable mass on both sides', () => {
    const both = direct(['swisscom', Number.NaN], ['swisscomm', Number.NaN], {
      alpha: 1,
      beta: 1,
      elementSimilarity: SOFT,
    })
    const clean = direct(['swisscom'], ['swisscomm'], {
      alpha: 1,
      beta: 1,
      elementSimilarity: SOFT,
    })
    expect(both).toBeLessThan(clean)
  })

  it('ignores a weightless element entirely', () => {
    const elementWeights = new Map([
      ['swisscom', 1],
      ['ag', 0],
    ])
    expect(
      direct(['swisscom', 'ag'], ['swisscomm'], {
        elementWeights,
        elementSimilarity: SOFT,
      }),
    ).toBe(
      direct(['swisscom'], ['swisscomm'], { elementWeights, elementSimilarity: SOFT }),
    )
  })
})

describe('thresholds', () => {
  const a = ['swisscom', 'ag']
  const b = ['swisscomm', 'gmbh']

  it('reports a score that clears the cutoff', () => {
    const score = direct(a, b, { elementSimilarity: SOFT })
    expect(
      tverskySimilarity(a, b, unigram({ elementSimilarity: SOFT, scoreCutoff: score })),
    ).toBe(score)
  })

  it('reports zero for a one-shot score below the cutoff', () => {
    expect(
      tverskySimilarity(a, b, unigram({ elementSimilarity: SOFT, scoreCutoff: 0.99 })),
    ).toBe(0)
  })

  it('reports zero for a prepared score below the cutoff', () => {
    const kernel = prepareScorerOf(tverskySimilarity)(
      a,
      unigram({ elementSimilarity: SOFT }),
    )
    expect(kernel(b, 0.99)).toBe(0)
    expect(kernel(b, null)).toBe(direct(a, b, { elementSimilarity: SOFT }))
  })

  // A missed distance reports the top of the range rather than the bottom.
  it('reports the maximum for a prepared distance beyond the cutoff', () => {
    const kernel = prepareScorerOf(tverskyDistance)(
      a,
      unigram({ elementSimilarity: SOFT }),
    )
    expect(kernel(b, 0.001)).toBe(1)
    expect(kernel(b, null)).toBe(1 - direct(a, b, { elementSimilarity: SOFT }))
  })
})

describe('a query with no weighted mass at all', () => {
  // What a prepared choice must keep on top of its counts. Zero-weight elements
  // are excluded from the distinct-element view — nothing can match on them —
  // yet all-zero weights are decided by multiset equality over exactly those
  // elements, so a choice that kept counts alone would answer this wrong on
  // every path but the one-shot one.
  const zero = { defaultElementWeight: 0, elementSimilarity: SOFT }

  it.each(PATHS)('proves equality by multiset through the %s path', (_name, score) => {
    expect(score(['alpha', 'beta'], ['alpha', 'beta'], zero)).toBe(1)
    expect(score(['alpha', 'beta'], ['alpha', 'gamma'], zero)).toBe(0)
    expect(score(['alpha', 'alpha'], ['alpha'], zero)).toBe(0)
    // Unmatchable on both sides is still unmatchable: two `NaN` occurrences are
    // not proof of anything, whatever they weigh.
    expect(score(['alpha', Number.NaN], ['alpha', Number.NaN], zero)).toBe(0)
  })

  it.each(PATHS)(
    'scores a zero-weight query as the exact scorer does (%s)',
    (_name, score) => {
      const pairs: ReadonlyArray<readonly [readonly unknown[], readonly unknown[]]> = [
        [
          ['alpha', 'beta'],
          ['alpha', 'beta'],
        ],
        [['alpha'], ['alphaa']],
        [[], []],
      ]
      for (const [a, b] of pairs) {
        expect(score(a, b, zero)).toBe(score(a, b, { defaultElementWeight: 0 }))
      }
    },
  )
})

describe('extreme weights', () => {
  it('drops an edge whose shared mass underflows to nothing', () => {
    // Both tokens weigh 1e-300 and are 1e-30 alike, so the mass they could
    // share is 1e-330 — below the smallest subnormal, and therefore no mass at
    // all. Crediting a zero-mass pairing would only widen the tie plateau.
    const scorer = pairsOnly([['alpha', 'beta', 1e-30]])
    const configuration = {
      elementWeights: new Map([
        ['alpha', 1e-300],
        ['beta', 1e-300],
      ]),
      elementSimilarity: { scorer, threshold: 1e-30 },
    }
    expect(direct(['alpha'], ['beta'], configuration)).toBe(
      direct(['alpha'], ['beta'], { elementWeights: configuration.elementWeights }),
    )
  })

  it('stays in 0..1 with coefficients at the top of the range', () => {
    const score = direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
      alpha: Number.MAX_VALUE,
      beta: Number.MAX_VALUE,
      elementSimilarity: SOFT,
    })
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
    expect(Number.isFinite(score)).toBe(true)
  })

  it('stays in 0..1 with element weights at the top of the range', () => {
    const score = direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
      alpha: 1,
      beta: 1,
      elementWeights: new Map([
        ['swisscom', 1e300],
        ['swisscomm', 1e-300],
      ]),
      elementSimilarity: SOFT,
    })
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
    expect(Number.isFinite(score)).toBe(true)
  })

  // The residual fold is new arithmetic on old numbers: it forms `weight −
  // min(wa, wb) · s` per pair, which the exact path never computes. The exact
  // weighted suites therefore say nothing about it, and every corner of the
  // double range has to be walked here.
  const HOSTILE_WEIGHTS: ReadonlyArray<readonly [string, number]> = [
    ['the smallest subnormal', Number.MIN_VALUE],
    ['an epsilon', Number.EPSILON],
    ['a tiny normal', 1e-300],
    ['one', 1],
    ['a wide integer', 1e16],
    ['the top of the range', Number.MAX_VALUE],
  ]

  const HOSTILE_SIMILARITIES: ReadonlyArray<readonly [string, number]> = [
    ['exactly one', 1],
    ['one ulp below one', 1 - Number.EPSILON / 2],
    ['two ulps below one', 1 - Number.EPSILON],
    ['exactly at the threshold', 0.5],
    ['a hair above the threshold', 0.5 + Number.EPSILON],
  ]

  describe.each(HOSTILE_SIMILARITIES)('at a similarity of %s', (_label, similarity) => {
    const inner = pairsOnly([['heavy', 'light', similarity]])

    it.each(HOSTILE_WEIGHTS)(
      'holds its contract with the first element at %s',
      (_first, firstWeight) => {
        for (const [, secondWeight] of HOSTILE_WEIGHTS) {
          const elementWeights = new Map([
            ['heavy', firstWeight],
            ['light', secondWeight],
          ])
          const soften = {
            elementWeights,
            elementSimilarity: { scorer: inner, threshold: 0.5 },
          }
          // A few of these pairs span more than a weight table can hold —
          // `Number.MIN_VALUE` beside `Number.MAX_VALUE`. That is the exact
          // path's own rule, and element similarity may not soften it.
          let exact = 0
          try {
            exact = direct(['heavy'], ['light'], { elementWeights })
          } catch (refusal) {
            expect(() => direct(['heavy'], ['light'], soften)).toThrow(refusal)
            continue
          }
          const soft = direct(['heavy'], ['light'], soften)
          expect(Number.isFinite(soft)).toBe(true)
          expect(soft).toBeGreaterThanOrEqual(0)
          expect(soft).toBeLessThanOrEqual(1)
          expect(soft).toBeGreaterThanOrEqual(exact - 1e-12)
        }
      },
    )
  })
})

describe('configuration ownership', () => {
  // `scorer` and `threshold` are compiled into the scorer, and the JSDoc says so
  // out loud. A caller who keeps the options object and edits it later must not
  // be able to move a score.
  it('ignores a threshold raised after the scorer was created', () => {
    const option = { scorer: indel, threshold: 0.8 }
    const scorer = createScorer(tverskyMetric, {
      gramSize: 1,
      elementSimilarity: option,
    })
    const before = scorer.score(['swisscom', 'ag'], ['swisscomm', 'ag'])

    option.threshold = 0.99

    expect(scorer.score(['swisscom', 'ag'], ['swisscomm', 'ag'])).toBe(before)
    expect(before).not.toBe(direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {}))
  })

  it('ignores an inner scorer swapped after the scorer was created', () => {
    const option = { scorer: indel, threshold: 0.8 }
    const scorer = createScorer(tverskyMetric, {
      gramSize: 1,
      elementSimilarity: option,
    })
    const before = scorer.score(['swisscom', 'ag'], ['swisscomm', 'ag'])

    option.scorer = pairsOnly([['swisscom', 'swisscomm', 1]])

    expect(scorer.score(['swisscom', 'ag'], ['swisscomm', 'ag'])).toBe(before)
  })
})

describe('the soft engine and the exact engines stay apart', () => {
  it('refuses element similarity above gramSize 1', () => {
    expect(() => direct(['a'], ['b'], { gramSize: 2, elementSimilarity: SOFT })).toThrow(
      new RangeError('element similarity is only defined at gramSize 1'),
    )
  })

  it('refuses it at the default gram size', () => {
    expect(() =>
      createScorer(tverskyMetric, { elementSimilarity: SOFT }).score(['a'], ['b']),
    ).toThrow(new RangeError('element similarity is only defined at gramSize 1'))
  })

  it('never collapses a soft configuration to the shared default', () => {
    // The collapse needs `gramSize: 2`, which a soft configuration cannot have,
    // so a soft configuration never canonicalizes to the empty record and its
    // prepared choices can never reach an exact scorer's kernel.
    const soft = createScorer(tverskyMetric, unigram({ elementSimilarity: SOFT }))
    const rows = [{ prepared: soft.prepareChoice(['swisscom']) }]
    expect(
      bestMatch(['swisscom'], rows, { scorer: soft, getPrepared: (row) => row.prepared })
        ?.score,
    ).toBe(1)
    const options = {
      scorer: createScorer(tverskyMetric),
      getPrepared: () => rows[0].prepared,
    }
    expect(() =>
      Reflect.apply(bestMatch, undefined, [['swisscom'], rows, options]),
    ).toThrow('prepared choice is incompatible with this scorer')
  })

  // A weight table that prices nothing compiles away, and the soft option has
  // to survive that branch rather than being dropped with the weights.
  it('keeps element similarity when a uniform weight table compiles away', () => {
    const uniform = {
      elementWeights: new Map([
        ['swisscom', 2],
        ['ag', 2],
      ]),
      defaultElementWeight: 2,
    }
    expect(
      direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
        ...uniform,
        elementSimilarity: SOFT,
      }),
    ).toBe(direct(['swisscom', 'ag'], ['swisscomm', 'ag'], { elementSimilarity: SOFT }))
    // And the option really was reaching the engine.
    expect(
      direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
        ...uniform,
        elementSimilarity: SOFT,
      }),
    ).not.toBe(direct(['swisscom', 'ag'], ['swisscomm', 'ag'], uniform))
  })
})

describe('a soft scorer declines the guarantees it cannot make', () => {
  it('reports itself asymmetric even at equal weights', () => {
    const scorer = createScorer(
      tverskyMetric,
      unigram({ alpha: 1, beta: 1, elementSimilarity: SOFT }),
    )
    expect(scorer.symmetric).toBe(false)
    // `scoreMatrix` mirrors a symmetric scorer's cells rather than scoring
    // them, so every cell here is scored on its own.
    const items = [['swisscom', 'ag'], ['swisscomm']]
    const matrix = scoreMatrix(items, items, { scorer })
    expect(matrix.at(0, 1)).toBe(scorer.score(items[0], items[1]))
    expect(matrix.at(1, 0)).toBe(scorer.score(items[1], items[0]))
  })

  it('offers no indexed representation', () => {
    const scorer = createScorer(tverskyMetric, unigram({ elementSimilarity: SOFT }))
    expect(() => createIndexedMatcher([['swisscom']], { scorer })).toThrow(TypeError)
  })

  it('offers none in the distance direction either', () => {
    const scorer = createScorer(
      tverskyDistanceMetric,
      unigram({ elementSimilarity: SOFT }),
    )
    expect(scorer.direction).toBe('distance')
  })
})

describe('determinism', () => {
  it('gives the same bits on every call and every fresh scorer', () => {
    const a = ['swisscom', 'swisscoma', 'ag']
    const b = ['swisscomm', 'swisscomb', 'gmbh']
    const once = direct(a, b, { elementSimilarity: SOFT })
    expect(direct(a, b, { elementSimilarity: SOFT })).toBe(once)
    expect(configured(a, b, { elementSimilarity: SOFT })).toBe(once)
    expect(
      createScorer(tverskyMetric, unigram({ elementSimilarity: SOFT })).score(a, b),
    ).toBe(once)
  })
})

describe('an inner scorer that is itself a Tversky scorer', () => {
  // The weighted paths share a module-level scratch array documented as never
  // being two calls deep; a soft score calls an arbitrary scorer mid-fold.
  it('does not disturb the outer weighted fold', () => {
    const innerTversky = createScorer(tverskyMetric, {
      gramSize: 1,
      alpha: 2,
      beta: 2,
      elementWeights: new Map([
        ['s', 4],
        ['w', 0.5],
      ]),
    })
    const configuration = {
      alpha: 1,
      beta: 1,
      elementWeights: new Map([
        ['swisscom', 5],
        ['ag', 0.1],
      ]),
      elementSimilarity: { scorer: innerTversky, threshold: 0.3 },
    }
    const withTversky = direct(['swisscom', 'ag'], ['swisscomm', 'ag'], configuration)
    const equivalent = pairsOnly([
      ['swisscom', 'swisscomm', innerTversky.score('swisscom', 'swisscomm')],
    ])
    expect(
      direct(['swisscom', 'ag'], ['swisscomm', 'ag'], {
        ...configuration,
        elementSimilarity: { scorer: equivalent, threshold: 0.3 },
      }),
    ).toBe(withTversky)
  })
})

describe('the size limit', () => {
  function tokens(count: number, prefix: string): string[] {
    return Array.from({ length: count }, (_unused, at) => `${prefix}-token-${at}`)
  }

  it('refuses a pair with more distinct leftovers on a side than the limit', () => {
    expect(() =>
      direct(tokens(33, 'left'), tokens(32, 'right'), { elementSimilarity: SOFT }),
    ).toThrow(RangeError)
  })

  it('refuses it on the second side too', () => {
    expect(() =>
      direct(tokens(32, 'left'), tokens(33, 'right'), { elementSimilarity: SOFT }),
    ).toThrow(RangeError)
  })

  it('allows the limit exactly', () => {
    expect(() =>
      direct(tokens(32, 'left'), tokens(32, 'right'), { elementSimilarity: SOFT }),
    ).not.toThrow()
  })

  // The limit is per side rather than on the product, because the solve is what
  // a long side against a short one makes expensive: the matching runs a
  // shortest-path search over `n + m` nodes once per augmentation, and 1 x 1024
  // costs 1081 M node visits where 32 x 32 costs 0.14 M.
  it('refuses a long sequence however short the other side is', () => {
    expect(() =>
      direct(tokens(512, 'left'), tokens(2, 'right'), { elementSimilarity: SOFT }),
    ).toThrow(RangeError)
  })

  // Occurrences are not distinct elements, and only distinct elements reach the
  // solver. Counting occurrences would refuse this pair, which costs one column
  // of the matrix and a single supply of 1024.
  it('counts a repeated token once however often it occurs', () => {
    const repeated = Array.from({ length: 1024 }, () => 'left-token-0')
    expect(() =>
      direct(repeated, tokens(32, 'right'), { elementSimilarity: SOFT }),
    ).not.toThrow()
  })

  // Counting every distinct leftover rather than the comparable ones would
  // refuse a pair that costs nothing at all.
  it('ignores leftovers no element scorer can see', () => {
    const objects = Array.from({ length: 300 }, () => ({}))
    const others = Array.from({ length: 300 }, () => ({}))
    expect(() => direct(objects, others, { elementSimilarity: SOFT })).not.toThrow()
  })

  it('costs nothing when one side has no comparable leftover', () => {
    expect(() =>
      direct(
        tokens(300, 'left'),
        Array.from({ length: 3 }, () => ({})),
        {
          elementSimilarity: SOFT,
        },
      ),
    ).not.toThrow()
  })
})
