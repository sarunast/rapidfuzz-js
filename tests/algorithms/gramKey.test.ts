import { describe, expect, it } from 'vitest'

import { feasibleRadices } from '../../src/algorithms/shared/gramKey.js'

describe('the radix ladder', () => {
  it('stays narrowest first, which is the order an index widens through', () => {
    // `NGramIndexBuilder` takes `feasibleRadices(gramSize)[0]` and re-keys
    // upward from it, so the order is load-bearing rather than cosmetic.
    expect(feasibleRadices(1)).toEqual([0x100, 0x1_0000, 0x11_0000])
    expect(feasibleRadices(3)).toEqual([0x100, 0x1_0000])
    expect(feasibleRadices(6)).toEqual([0x100])
    expect(feasibleRadices(7)).toEqual([])
  })
})
