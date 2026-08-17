// The element vocabulary itself. What it produces is covered wherever an index
// is built from arbitrary elements; what is pinned here is the rule the two
// re-key passes rely on — the second one may read the vocabulary and may not
// grow it.
import { describe, expect, it } from 'vitest'

import { ordinalizeChoice, resolveOrdinals, UNMATCHABLE } from './ordinals.js'

describe('resolving ordinals against a frozen vocabulary', () => {
  it('reads the ordinals an earlier pass assigned', () => {
    const table = new Map<unknown, number>()
    const assigned: number[] = []
    ordinalizeChoice(['a', 'b', 'a'], 2, table, assigned)
    const resolved: number[] = []
    resolveOrdinals(['a', 'b', 'a'], table, resolved)
    expect(resolved).toEqual(assigned)
  })

  it('refuses an element the vocabulary never saw', () => {
    // The radix an index re-keys with is sized from the finished vocabulary, so
    // an ordinal assigned after that point could be a digit it cannot hold.
    const table = new Map<unknown, number>([['a', 0]])
    expect(() => resolveOrdinals(['a', 'b'], table, [])).toThrow(
      'this element has no ordinal',
    )
  })

  it('leaves an element that cannot reach a gram out of the vocabulary', () => {
    const table = new Map<unknown, number>()
    const output: number[] = []
    ordinalizeChoice(['dead', NaN, 'a', 'b'], 2, table, output)
    expect(output).toEqual([UNMATCHABLE, UNMATCHABLE, 0, 1])
    expect([...table.keys()]).toEqual(['a', 'b'])
  })
})
