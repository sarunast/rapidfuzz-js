import { describe, expect, it } from 'vitest'

import { sharedFrequency } from './compare.js'
import { sharedFrequencyKernel } from './kernel.js'
import { buildProfile, NGramProfile } from './profile.js'

describe('the bounded kernel', () => {
  it('carries a suffix total past 2^31 without turning it negative', () => {
    // A sequence may be 0xffff_ffff long, so the frequency still to come can
    // pass 2^31. Held signed it wraps negative, the bound reads as unreachable
    // and the kernel abandons a candidate that shares everything it needs.
    // Built here rather than measured: the sequence itself would not fit.
    const counts = new Map<unknown, number>([
      ['common', 0x8000_0000],
      ['rare', 1],
    ])
    const query = new NGramProfile(
      1,
      0x8000_0001,
      0,
      { kind: 'trie', root: { children: null, counts } },
      null,
    )
    const choice = buildProfile(['rare'], 1)

    expect(sharedFrequencyKernel(query)(choice, 1)).toBe(1)
  })

  it('stops early only once the minimum is genuinely out of reach', () => {
    for (const gramSize of [1, 2, 3, 4]) {
      const query = buildProfile('abcabcab', gramSize)
      const kernel = sharedFrequencyKernel(query)
      const exact = sharedFrequency(query, buildProfile('abcabc', gramSize))
      // Asking for no more than the intersection holds must not cut the walk.
      expect(
        kernel(buildProfile('abcabc', gramSize), exact),
        `gramSize ${gramSize}`,
      ).toBe(exact)
      expect(kernel(buildProfile('abcabc', gramSize), 0), `gramSize ${gramSize}`).toBe(
        exact,
      )
      // Above it the answer may be any count below the minimum, never at it.
      expect(
        kernel(buildProfile('abcabc', gramSize), exact + 1),
        `gramSize ${gramSize}`,
      ).toBeLessThan(exact + 1)
    }
  })
})
