import { describe, expect, it } from 'vitest'

import type { ChoiceTable, ChoiceTableBuild } from '../../src/search/choiceTable.js'
import { buildChoiceTable, keyAt, matchAt } from '../../src/search/choiceTable.js'
import type { Items } from '../../src/search/types.js'

/** Keeps everything but the literal `'skip'`. */
function tableOf(items: Items<string>): ChoiceTableBuild<string, string> {
  return buildChoiceTable(items, (item: string) => (item === 'skip' ? null : item))
}

function keysOf(table: ChoiceTable<string>): unknown[] {
  return table.items.map((_, id) => keyAt(table, id))
}

describe('choice table', () => {
  it('leaves keys unmaterialized for a gapless array', () => {
    const { table, values } = tableOf(['a', 'b', 'c'])
    expect(table.keys).toBeNull()
    expect(table.items).toEqual(['a', 'b', 'c'])
    expect(keysOf(table)).toEqual([0, 1, 2])
    // The values contract: one entry per retained choice, in ascending id order.
    expect(values).toEqual(['a', 'b', 'c'])
  })

  it('materializes keys as soon as the first choice is skipped', () => {
    const { table, values } = tableOf(['skip', 'b', 'c'])
    expect(table.keys).toEqual([1, 2])
    expect(keysOf(table)).toEqual([1, 2])
    expect(values).toEqual(['b', 'c'])
  })

  it('keeps the source key of a choice after a gap', () => {
    const { table } = tableOf(['a', 'skip', 'c'])
    expect(table.keys).toEqual([0, 2])
    expect(matchAt(table, 1, 0.5)).toEqual({ item: 'c', key: 2, score: 0.5 })
  })

  it('back-fills the keys a divergence later than the first choice', () => {
    // 0 and 1 are their own positions, so nothing is stored until 7 arrives at
    // id 2 — the back-fill is the only thing that keeps ids 0 and 1 addressable.
    const { table } = tableOf(
      new Map([
        [0, 'a'],
        [1, 'b'],
        [7, 'c'],
      ]),
    )
    expect(table.keys).toEqual([0, 1, 7])
    expect(keysOf(table)).toEqual([0, 1, 7])
  })

  it('keeps object keys as the strings they are', () => {
    const { table } = tableOf({ 0: 'a', 1: 'b' })
    // `'0' !== 0`, so the first property already diverges from its position.
    expect(table.keys).toEqual(['0', '1'])
    expect(keyAt(table, 0)).toBe('0')
  })
})
