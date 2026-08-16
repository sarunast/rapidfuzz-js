// Not ported from RapidFuzz — the growth policy below is ours.
//
// A span-backed build either doubles its mask storage and pays for the copies,
// or takes the whole symbol span at once and pays for the waste. `grownMasks`
// takes the span past `MAX_SPAN_BLOCKS`, so these run either side of that: what
// must not change with the policy is the alignment it produces, which is what
// makes the next attempt at the trade cheap to check.
//
// The pattern shape is the one the choice matters for — a span as wide as the
// input with far fewer distinct elements in it, which is what makes taking the
// span whole a 19.5x over-allocation rather than the right guess.
import { describe, expect, it } from 'vitest'

import { indelDistance } from '../indel/implementation.js'
import { lcsSeqEditops } from '../lcs/implementation.js'
import { levenshteinEditops } from '../levenshtein/editops.js'
import { levenshteinDistance } from '../levenshtein/metric.js'

const LENGTH = 4000

/** `distinct` values spread across a span the width of the input. */
function spread(distinct: number, seed: number): number[] {
  const step = Math.floor(LENGTH / distinct)
  return Array.from({ length: LENGTH }, (_, i) => ((i * 7 + seed) % distinct) * step)
}

// Either side of the 256 blocks the first growth happens at, so one is served
// by the initial allocation and the other by the span.
const bands = [
  { name: 'inside the blocks it started with', distinct: 200 },
  { name: 'taking the whole span to grow', distinct: 600 },
]

describe('mask growth either side of the span jump', () => {
  for (const { name, distinct } of bands) {
    it(`aligns exactly while ${name}`, () => {
      const a = spread(distinct, 0)
      const b = spread(distinct, 3)

      // An LCS edit script is inserts and deletes only, so its length is the
      // Indel distance; Levenshtein's is its own. Either one disagreeing means
      // the masks the policy allocated were not the masks that got filled.
      expect(lcsSeqEditops(a, b).operations.length).toBe(indelDistance(a, b))
      expect(levenshteinEditops(a, b).operations.length).toBe(levenshteinDistance(a, b))
    })
  }
})
