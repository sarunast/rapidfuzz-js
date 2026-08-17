import { describe, expect, test } from 'vitest'

import { grownShareCapacity } from './weightedOverlap.js'

// The traversal itself is covered where its answers are — against the exhaustive
// weighted scorer, in `weightedTversky.test.ts`. What cannot be reached that way
// is the boundary a growth step holds to, since the corpus that would reach it
// cannot be allocated: the arithmetic is checked here instead.
describe('the share entry capacity', () => {
  test('doubles from a first block up to the last addressable entry', () => {
    expect(grownShareCapacity(0)).toBe(64)
    expect(grownShareCapacity(64)).toBe(128)
    expect(grownShareCapacity(1 << 20)).toBe(1 << 21)
    // Doubling would ask for 2³², which a one-based reference cannot name.
    expect(grownShareCapacity(2 ** 31)).toBe(0xffff_ffff)
  })

  test('refuses to grow past what a one-based reference can name', () => {
    expect(() => grownShareCapacity(0xffff_ffff)).toThrow(
      'cannot read more than 4294967295 shared postings',
    )
  })
})
