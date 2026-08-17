// The soft engine's own contracts: the exact reservation it hands the solver,
// the prepared-choice guard, and the `null` that means "nothing was matched
// fuzzily, so return the exact answer untouched".

import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'

import { normalizedSimilarity as indelSimilarity } from '../indel/index.js'
import {
  compileElementSimilarity,
  ElementKernels,
  elementScore,
  preparedElementScore,
} from './elementSimilarity.js'
import { occurrencesOf } from './occurrences.js'
import {
  preparedSoftChoice,
  softComponentsOf,
  SoftTverskyChoice,
  softTablesOf,
} from './soft.js'

const soft = compileElementSimilarity(
  { scorer: createScorer(indelSimilarity), threshold: 0.8 },
  1,
)

function tablesOf(first: readonly unknown[], second: readonly unknown[]) {
  return softTablesOf(occurrencesOf(first, null), occurrencesOf(second, null))
}

describe('softTablesOf', () => {
  it('reserves every occurrence exact matching can claim', () => {
    const tables = tablesOf(['react', 'react', 'vue'], ['react', 'angular'])
    expect(tables.overlap.sharedCount).toBe(1)
    expect([...tables.overlap.leftoverFirst]).toEqual([1, 1])
    expect([...tables.overlap.leftoverSecond]).toEqual([0, 1])
  })

  it('leaves everything over when nothing is equal', () => {
    const tables = tablesOf(['alpha'], ['beta'])
    expect(tables.overlap.sharedCount).toBe(0)
    expect([...tables.overlap.leftoverFirst]).toEqual([1])
  })

  it('counts an element the other side does not hold at all', () => {
    const tables = tablesOf(['alpha'], [])
    expect(tables.overlap.sharedCount).toBe(0)
    expect([...tables.overlap.leftoverFirst]).toEqual([1])
    expect([...tables.overlap.leftoverSecond]).toEqual([])
  })
})

describe('softComponentsOf is null where nothing was matched fuzzily', () => {
  const EMPTY: ReadonlyArray<readonly [string, readonly unknown[], readonly unknown[]]> =
    [
      ['one side has no leftover', ['react'], ['react']],
      ['one side is empty', ['react'], []],
      ['no leftover is comparable', [97], [98]],
      ['nothing reaches the threshold', ['alpha'], ['beta']],
    ]

  it.each(EMPTY)('returns null when %s', (_label, first, second) => {
    const tables = tablesOf(first, second)
    expect(softComponentsOf(tables, soft, tables.overlap.sharedCount, null)).toBeNull()
  })

  it('returns components once an edge survives', () => {
    const tables = tablesOf(['swisscom'], ['swisscomm'])
    const components = softComponentsOf(tables, soft, tables.overlap.sharedCount, null)
    expect(components).not.toBeNull()
    expect(components?.edges).toHaveLength(1)
    expect([...(components?.units ?? [])]).toEqual([1])
  })

  it('adds only to the shared mass it was given', () => {
    const tables = tablesOf(['react', 'swisscom'], ['react', 'swisscomm'])
    const components = softComponentsOf(tables, soft, tables.overlap.sharedCount, null)
    expect(components?.shared).toBeGreaterThan(tables.overlap.sharedCount)
  })
})

describe('held query kernels', () => {
  // One query against many choices, which is the only lifetime a cache has: it
  // belongs to a prepared query, so the first side is fixed while the second
  // changes. The corpus is long enough for the cache to promote partway
  // through, so both the direct and the prepared halves of the policy are here.
  const QUERY = ['swisscom', 'holding', 'react']
  const CHOICES: readonly (readonly unknown[])[] = [
    ['swisscomm', 'holdings', 'reactt'],
    ['nothing', 'alike', 'atall'],
    ['react', 'swisscomm'],
    ['holdingg', 'holdingg', 'swisscom'],
    ['swisscomm'],
    ['reacts', 'holdings', 'swisscomm', 'swisscomm'],
    [97, 98, 'swisscomm'],
    ['swisscom', 'holding', 'react'],
  ]

  it('scores every candidate exactly as an unheld pair does', () => {
    const kernels = new ElementKernels(soft)
    for (const choice of CHOICES) {
      const direct = softComponentsOf(tablesOf(QUERY, choice), soft, 0, null)
      const held = softComponentsOf(tablesOf(QUERY, choice), soft, 0, kernels)
      expect(held?.shared).toBe(direct?.shared)
      expect(held?.firstOnly).toBe(direct?.firstOnly)
      expect(held?.secondOnly).toBe(direct?.secondOnly)
      expect([...(held?.units ?? [])]).toEqual([...(direct?.units ?? [])])
      expect(held?.edges).toEqual(direct?.edges)
    }
  })

  it('compares directly until a scan has earned the preparation', () => {
    const kernels = new ElementKernels(soft)
    expect(kernels.earned(3)).toBe(false)
    expect(kernels.earned(4)).toBe(false)
    // The pair that reaches the threshold is counted but still scored directly;
    // only what comes after it can consume a kernel.
    expect(kernels.earned(1)).toBe(false)
    expect(kernels.earned(1)).toBe(true)
  })

  it('refills one array of columns rather than allocating per pair', () => {
    const kernels = new ElementKernels(soft)
    const columns = kernels.columnsFor([{ operand: 'alpha' }, { operand: 'beta' }])
    expect(columns).toHaveLength(2)
    const next = kernels.columnsFor([{ operand: 'gamma' }])
    expect(next).toBe(columns)
    expect(next).toHaveLength(1)
    // Opaque, but still what a kernel consumes — which is the only property of
    // them this owns; what a prepared choice *is* belongs to the element scorer.
    expect(preparedElementScore(soft, kernels.kernelFor('gamma'), next[0])).toBe(1)
  })
})

describe('preparedElementScore carries the element scorer onto 0..1', () => {
  const CASES: ReadonlyArray<
    readonly [string, readonly [number, number], (a: unknown, b: unknown) => number]
  > = [
    ['a 0..100 scorer', [0, 100], (a, b) => (a === b ? 100 : 80)],
    ['a shifted -1..1 scorer', [-1, 1], (a, b) => (a === b ? 1 : 0.6)],
  ]

  it.each(CASES)('rescales %s as the direct path does', (_label, bounds, metric) => {
    const compiled = compileElementSimilarity(
      {
        scorer: createScorer(metric, {
          direction: 'similarity',
          bounds,
          symmetric: true,
        }),
        threshold: 0.8,
      },
      1,
    )
    const kernels = new ElementKernels(compiled)
    const kernel = kernels.kernelFor('alpha')
    const columns = kernels.columnsFor([{ operand: 'alpha' }, { operand: 'beta' }])
    expect(preparedElementScore(compiled, kernel, columns[0])).toBe(1)
    expect(preparedElementScore(compiled, kernel, columns[1])).toBe(
      elementScore(compiled, 'alpha', 'beta'),
    )
  })

  it('keeps an edge that sits exactly on the threshold, and drops the one below', () => {
    const onThreshold = compileElementSimilarity(
      {
        scorer: createScorer((a, b) => (a === b ? 100 : 80), {
          direction: 'similarity',
          bounds: [0, 100],
          symmetric: true,
        }),
        threshold: 0.8,
      },
      1,
    )
    // Earned up front: a one-element pair would otherwise be compared directly,
    // and the boundary this pins is the prepared path's.
    const kernels = new ElementKernels(onThreshold)
    kernels.earned(1024)
    const held = softComponentsOf(tablesOf(['alpha'], ['beta']), onThreshold, 0, kernels)
    expect(held?.edges.map((edge) => edge.similarity)).toEqual([0.8])
    const justUnder = compileElementSimilarity(
      {
        scorer: createScorer((a, b) => (a === b ? 100 : 80), {
          direction: 'similarity',
          bounds: [0, 100],
          symmetric: true,
        }),
        threshold: 0.8000001,
      },
      1,
    )
    const under = new ElementKernels(justUnder)
    under.earned(1024)
    expect(
      softComponentsOf(tablesOf(['alpha'], ['beta']), justUnder, 0, under),
    ).toBeNull()
  })
})

describe('preparedSoftChoice', () => {
  it('accepts a choice this engine prepared', () => {
    const choice = new SoftTverskyChoice(occurrencesOf(['react'], null))
    expect(preparedSoftChoice(choice)).toBe(choice)
  })

  it.each([
    ['a bare sequence', ['react']],
    ['a string', 'react'],
    ['null', null],
    ['a foreign object', { occurrences: [] }],
  ])('refuses %s', (_label, value) => {
    expect(() => preparedSoftChoice(value)).toThrow(
      new TypeError('invalid prepared soft tversky choice'),
    )
  })
})
