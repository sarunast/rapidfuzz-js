// The soft engine's own contracts: the exact reservation it hands the solver,
// the prepared-choice guard, and the `null` that means "nothing was matched
// fuzzily, so return the exact answer untouched".

import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'

import { normalizedSimilarity as indelSimilarity } from '../indel/index.js'
import { compileElementSimilarity } from './elementSimilarity.js'
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
    expect(softComponentsOf(tables, soft, tables.overlap.sharedCount)).toBeNull()
  })

  it('returns components once an edge survives', () => {
    const tables = tablesOf(['swisscom'], ['swisscomm'])
    const components = softComponentsOf(tables, soft, tables.overlap.sharedCount)
    expect(components).not.toBeNull()
    expect(components?.edges).toHaveLength(1)
    expect([...(components?.units ?? [])]).toEqual([1])
  })

  it('adds only to the shared mass it was given', () => {
    const tables = tablesOf(['react', 'swisscom'], ['react', 'swisscomm'])
    const components = softComponentsOf(tables, soft, tables.overlap.sharedCount)
    expect(components?.shared).toBeGreaterThan(tables.overlap.sharedCount)
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
