import { describe, expect, test } from 'vitest'

import {
  impossibleTrustedThreshold,
  qualifies,
  trustedKernelThreshold,
  validateThreshold,
} from '../../src/core/threshold.js'

// Every assertion is an exact boundary. The two functions differ only in strict
// versus inclusive comparisons, and a `>` that becomes a `>=` skips an
// algorithm the caller wanted — which nothing else in the suite would notice.
describe('threshold boundaries', () => {
  test('a similarity threshold is impossible only above the upper bound', () => {
    expect(impossibleTrustedThreshold('similarity', [0, 100], 100)).toBe(false)
    expect(impossibleTrustedThreshold('similarity', [0, 100], 100.0001)).toBe(true)
    expect(impossibleTrustedThreshold('similarity', [0, 100], 0)).toBe(false)
    expect(impossibleTrustedThreshold('similarity', [0, 100], -1)).toBe(false)
    expect(impossibleTrustedThreshold('similarity', [0, 1], 1 + Number.EPSILON)).toBe(
      true,
    )
    expect(impossibleTrustedThreshold('similarity', [0, 100], null)).toBe(false)
  })

  test('a distance threshold is impossible only below the lower bound', () => {
    expect(impossibleTrustedThreshold('distance', [0, 100], 0)).toBe(false)
    expect(impossibleTrustedThreshold('distance', [0, 100], -0.0001)).toBe(true)
    expect(impossibleTrustedThreshold('distance', [0, 100], 100)).toBe(false)
    // No finite maximum, so no finite threshold nothing can reach.
    expect(impossibleTrustedThreshold('distance', [0, Infinity], 1e100)).toBe(false)
    expect(impossibleTrustedThreshold('distance', [0, Infinity], -1)).toBe(true)
    expect(impossibleTrustedThreshold('distance', [0, 100], null)).toBe(false)
  })

  test('a similarity cutoff falls away at or below the lower bound', () => {
    expect(trustedKernelThreshold('similarity', [0, 100], 0)).toBeNull()
    expect(trustedKernelThreshold('similarity', [0, 100], -1)).toBeNull()
    expect(trustedKernelThreshold('similarity', [0, 100], Number.MIN_VALUE)).toBe(
      Number.MIN_VALUE,
    )
    expect(trustedKernelThreshold('similarity', [0, 100], 0.0001)).toBe(0.0001)
    expect(trustedKernelThreshold('similarity', [0, 100], null)).toBeNull()
    // Impossible thresholds are settled before this, and come back unchanged.
    expect(trustedKernelThreshold('similarity', [0, 100], 101)).toBe(101)
  })

  test('a distance cutoff falls away at or above the upper bound', () => {
    expect(trustedKernelThreshold('distance', [0, 100], 100)).toBeNull()
    expect(trustedKernelThreshold('distance', [0, 100], 101)).toBeNull()
    expect(trustedKernelThreshold('distance', [0, 100], 99.999)).toBe(99.999)
    expect(trustedKernelThreshold('distance', [0, 1], 1 - Number.EPSILON / 2)).toBe(
      1 - Number.EPSILON / 2,
    )
    // An unbounded scorer never loses its cutoff, however large.
    expect(trustedKernelThreshold('distance', [0, Infinity], 1e100)).toBe(1e100)
    expect(trustedKernelThreshold('distance', [0, 100], null)).toBeNull()
  })

  test('qualifying reads the direction, and a threshold is finite', () => {
    expect(qualifies('similarity', 60, 60)).toBe(true)
    expect(qualifies('similarity', 59.9999, 60)).toBe(false)
    expect(qualifies('distance', 3, 3)).toBe(true)
    expect(qualifies('distance', 3.0001, 3)).toBe(false)
    for (const value of [Number.NaN, Infinity, -Infinity]) {
      expect(() => validateThreshold(value)).toThrow('threshold must be finite')
    }
    expect(validateThreshold(-0)).toBe(-0)
    expect(validateThreshold(1e308)).toBe(1e308)
  })
})
