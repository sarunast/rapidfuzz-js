// Not ported from RapidFuzz — these are edges the upstream suite does not
// cover, but whose answers were taken from it. Expected values verified against
// rapidfuzz 3.14.5 (the C++ path) on 2026-08-08:
//
//   partial_ratio('', 'x'*200)            -> 0.0
//   partial_ratio_alignment('', 'x'*200)  -> (0.0, 0, 0, 0, 0)
//   partial_ratio('', '')                 -> 100.0
//   partial_ratio_alignment('', '')       -> (100.0, 0, 0, 0, 0)
//   partial_ratio('x'*200, '')            -> 0.0
//
// The empty needle is the one that had teeth. `partialRatioScan` sizes its
// interior search from the haystack, so an empty pattern against a long text
// used to allocate a `Uint32Array` as long as the text and bisect it, computing
// `1 - distance / (2 * len1)` as `0 / 0` at every step. The resulting `NaN`
// failed every comparison, so the score came out right by accident. The guard in
// `partialRatioScan` makes it right on purpose, and these assertions keep it so.
import { describe, expect, it } from 'vitest'

import { partialRatio, partialRatioAlignment } from '../src/fuzz.js'

const LONG = 'x'.repeat(200)

describe('partialRatio with an empty input', () => {
  it('scores an empty needle against a long haystack as 0', () => {
    expect(partialRatio('', LONG)).toBe(0)
    expect(partialRatio(LONG, '')).toBe(0)
  })

  it('reports the degenerate alignment for an empty needle', () => {
    expect(partialRatioAlignment('', LONG)).toEqual({
      score: 0,
      srcStart: 0,
      srcEnd: 0,
      destStart: 0,
      destEnd: 0,
    })
  })

  it('scores two empty inputs as a perfect match', () => {
    expect(partialRatio('', '')).toBe(100)
    expect(partialRatioAlignment('', '')).toEqual({
      score: 100,
      srcStart: 0,
      srcEnd: 0,
      destStart: 0,
      destEnd: 0,
    })
  })

  // The interior search only engages past 64 windows, so this is the length that
  // would have taken the bisection with a `NaN` bound.
  it('does not depend on how long the haystack is', () => {
    for (const length of [1, 63, 64, 65, 200, 5000]) {
      expect(partialRatio('', 'y'.repeat(length))).toBe(0)
    }
  })

  it('still answers null when a cutoff rejects the empty alignment', () => {
    expect(partialRatioAlignment('', LONG, { scoreCutoff: 1 })).toBeNull()
  })
})
