// `explain` under `elementSimilarity`: which occurrence was paired with which,
// how alike they were, and the contract that the rows never move a total.

import { describe, expect, it } from 'vitest'

import { createScorer, type Scorer } from '#core/scoring/scorer.js'

import { normalizedSimilarity as indelSimilarity } from '../indel/index.js'
import { similarity as tverskyMetric } from './index.js'

const indel = createScorer(indelSimilarity)

interface SoftConfiguration {
  readonly alpha?: number
  readonly beta?: number
  readonly elementWeights?: ReadonlyMap<unknown, number>
  readonly defaultElementWeight?: number
  readonly elementSimilarity?: { scorer: Scorer<'similarity'>; threshold: number }
}

function explainer(configuration: SoftConfiguration) {
  return createScorer(tverskyMetric, { ...configuration, gramSize: 1 })
}

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

describe('a fuzzy pair is a match row, not two unmatched ones', () => {
  const scorer = pairsOnly([['swisscom', 'swisscomm', 0.75]])
  const evidence = explainer({
    alpha: 1,
    beta: 1,
    elementSimilarity: { scorer, threshold: 0.5 },
  }).explain(['swisscom', 'ag'], ['swisscomm', 'ag'])

  it('reports both pairs, ascending by first index', () => {
    expect(evidence.matches.map((match) => match.firstIndex)).toEqual([0, 1])
    expect(evidence.matches.map((match) => match.exact)).toEqual([false, true])
  })

  it('carries the element scorer’s similarity and each side’s own weight', () => {
    const fuzzy = evidence.matches[0]
    expect(fuzzy.first).toBe('swisscom')
    expect(fuzzy.second).toBe('swisscomm')
    expect(fuzzy.similarity).toBe(0.75)
    expect(fuzzy.firstWeight).toBe(1)
    expect(fuzzy.secondWeight).toBe(1)
    expect(fuzzy.sharedMass).toBe(0.75)
    expect(fuzzy.firstUnmatchedMass).toBe(0.25)
    expect(fuzzy.secondUnmatchedMass).toBe(0.25)
  })

  it('leaves a partially matched occurrence out of the unmatched arrays', () => {
    expect(evidence.unmatchedFirst).toEqual([])
    expect(evidence.unmatchedSecond).toEqual([])
  })

  it('keeps an exact row exact', () => {
    const exact = evidence.matches[1]
    expect(exact.exact).toBe(true)
    expect(exact.similarity).toBe(1)
    expect(exact.firstUnmatchedMass).toBe(0)
  })
})

describe('occurrences with no partner at all', () => {
  it('stay in the unmatched arrays', () => {
    const evidence = explainer({ elementSimilarity: SOFT }).explain(
      ['swisscom', 'orphan-token'],
      ['swisscomm'],
    )
    expect(evidence.matches.map((match) => match.first)).toEqual(['swisscom'])
    expect(evidence.unmatchedFirst.map((one) => one.element)).toEqual(['orphan-token'])
    expect(evidence.unmatchedSecond).toEqual([])
  })

  it('include an element the scorer never sees', () => {
    const evidence = explainer({ elementSimilarity: SOFT }).explain(
      ['swisscom', 'a'],
      ['swisscomm'],
    )
    expect(evidence.unmatchedFirst.map((one) => one.element)).toEqual(['a'])
  })

  it('come back in ascending index however the elements were ordered', () => {
    const evidence = explainer({ elementSimilarity: SOFT }).explain(
      ['zeta-token', 'swisscom', 'alpha-token', Number.NaN],
      ['swisscomm'],
    )
    expect(evidence.unmatchedFirst.map((one) => one.index)).toEqual([0, 2, 3])
    expect(evidence.unmatchedFirst.map((one) => one.element)).toEqual([
      'zeta-token',
      'alpha-token',
      Number.NaN,
    ])
  })
})

describe('which occurrence a repeat gives up', () => {
  // Exact matching reserves the earliest, so the later one is what the element
  // scorer is offered.
  it('offers the later occurrence to the element scorer', () => {
    const scorer = pairsOnly([['react', 'reakt', 0.75]])
    const evidence = explainer({
      elementSimilarity: { scorer, threshold: 0.5 },
    }).explain(['react', 'react'], ['react', 'reakt'])
    expect(evidence.matches.map((match) => [match.firstIndex, match.exact])).toEqual([
      [0, true],
      [1, false],
    ])
    expect(evidence.matches[1].secondIndex).toBe(1)
  })
})

describe('the totals and the score', () => {
  const CASES: ReadonlyArray<readonly [string, readonly unknown[], readonly unknown[]]> =
    [
      ['a typo', ['swisscom', 'ag'], ['swisscomm', 'ag']],
      ['two typos', ['swisscom', 'gmbh'], ['swisscomm', 'gmbhh']],
      ['a repeat', ['react', 'react'], ['reakt']],
      ['nothing in common', ['alpha'], ['beta']],
      ['an orphan', ['swisscom', 'orphan-token'], ['swisscomm']],
      ['unmatchable mass', ['swisscom', Number.NaN], ['swisscomm']],
    ]

  it.each(CASES)('agrees with score() on %s', (_label, a, b) => {
    const scorer = explainer({ alpha: 1, beta: 0.1, elementSimilarity: SOFT })
    expect(scorer.explain(a, b).score).toBe(scorer.score(a, b))
  })

  it.each(CASES)('agrees with score() on %s when weighted', (_label, a, b) => {
    const scorer = explainer({
      alpha: 1,
      beta: 0.1,
      elementWeights: new Map<unknown, number>([
        ['swisscom', 5],
        ['swisscomm', 3],
        ['ag', 0.1],
      ]),
      elementSimilarity: SOFT,
    })
    expect(scorer.explain(a, b).score).toBe(scorer.score(a, b))
  })

  it('reports the shared mass the fold produced', () => {
    const scorer = pairsOnly([['swisscom', 'swisscomm', 0.75]])
    const evidence = explainer({
      alpha: 1,
      beta: 1,
      elementSimilarity: { scorer, threshold: 0.5 },
    }).explain(['swisscom'], ['swisscomm'])
    expect(evidence.totals.sharedMass).toBe(0.75)
    expect(evidence.totals.firstUnmatchedMass).toBe(0.25)
    expect(evidence.totals.secondUnmatchedMass).toBe(0.25)
    expect(evidence.similarity).toBe(0.75 / (0.75 + (0.25 + 0.25)))
  })

  it('keeps each side’s own mass untouched by fuzzy matching', () => {
    const soft = explainer({ elementSimilarity: SOFT }).explain(
      ['swisscom', 'ag'],
      ['swisscomm', 'ag'],
    )
    const exact = explainer({}).explain(['swisscom', 'ag'], ['swisscomm', 'ag'])
    expect(soft.totals.firstMass).toBe(exact.totals.firstMass)
    expect(soft.totals.secondMass).toBe(exact.totals.secondMass)
  })
})

describe('an unreachable threshold explains exactly what the exact scorer does', () => {
  const UNREACHABLE = { scorer: indel, threshold: 1 }

  const UNCHANGED: ReadonlyArray<
    readonly [string, readonly unknown[], readonly unknown[]]
  > = [
    ['a typo', ['swisscom', 'ag'], ['swisscomm', 'ag']],
    ['nothing in common', ['alpha'], ['beta']],
    ['both empty', [], []],
    ['unmatchable mass', ['swisscom', Number.NaN], ['swisscom']],
  ]

  it.each(UNCHANGED)('reports %s identically', (_label, a, b) => {
    expect(explainer({ elementSimilarity: UNREACHABLE }).explain(a, b)).toEqual(
      explainer({}).explain(a, b),
    )
  })
})
