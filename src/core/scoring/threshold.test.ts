import { describe, expect, test } from 'vitest'

import * as levenshtein from '../../algorithms/levenshtein/index.js'
import { bestMatch, createMatcher, search, searchIter } from '../../index.js'
import { createScorer, scorerCompilation } from './scorer.js'
import {
  impossibleThreshold,
  impossibleTrustedThreshold,
  kernelThreshold,
  optionalThreshold,
  passesThreshold,
  qualifies,
  trustedKernelThreshold,
  knownOptimum,
  validateThreshold,
} from './threshold.js'

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
    expect(optionalThreshold(undefined)).toBeNull()
    expect(optionalThreshold(60)).toBe(60)
    expect(() => optionalThreshold(Infinity)).toThrow('threshold must be finite')
  })

  // Same boundaries as `qualifies`, plus the one thing it cannot say: a caller
  // who asked for no threshold rejects nothing, in either direction.
  test('the nullable form agrees with the strict one, and null passes', () => {
    expect(passesThreshold('similarity', 60, 60)).toBe(true)
    expect(passesThreshold('similarity', 59.9999, 60)).toBe(false)
    expect(passesThreshold('distance', 3, 3)).toBe(true)
    expect(passesThreshold('distance', 3.0001, 3)).toBe(false)
    expect(passesThreshold('similarity', 0, null)).toBe(true)
    expect(passesThreshold('distance', Number.MAX_VALUE, null)).toBe(true)
  })
})

// The three helpers every search goes through, and the one rule they exist to
// state once: a custom scorer's bounds are the caller's claim rather than the
// algorithm's, so nothing may be concluded from them. Pinned here because the
// searches that would notice a wrong answer are the ones with a custom scorer
// and a threshold outside its declared bounds — a shape easy to leave untested
// at each of the six call sites this replaced.
describe('what a compilation concludes about a threshold', () => {
  const trusted = scorerCompilation(createScorer(levenshtein.normalizedSimilarity))
  const custom = scorerCompilation(
    createScorer(() => 1, { direction: 'similarity', bounds: [0, 1], symmetric: true }),
  )

  test('a trusted compilation applies its own bounds', () => {
    expect(impossibleThreshold(trusted, 1.5)).toBe(true)
    expect(impossibleThreshold(trusted, 1)).toBe(false)
    expect(impossibleThreshold(trusted, null)).toBe(false)
    expect(kernelThreshold(trusted, 0)).toBeNull()
    expect(kernelThreshold(trusted, 0.5)).toBe(0.5)
    expect(knownOptimum(trusted)).toBe(1)
  })

  test('a custom one concludes nothing, whatever bounds it declared', () => {
    expect(impossibleThreshold(custom, 1.5)).toBe(false)
    expect(impossibleThreshold(custom, 99)).toBe(false)
    // Handed through unchanged where a trusted scorer would have lost it.
    expect(kernelThreshold(custom, 0)).toBe(0)
    expect(kernelThreshold(custom, 0.5)).toBe(0.5)
    expect(kernelThreshold(custom, null)).toBeNull()
    // No score is known to be unbeatable, so no scan may stop early.
    expect(knownOptimum(custom)).toBeNull()
  })

  test('a trusted distance reads the other end of its bounds', () => {
    const distance = scorerCompilation(createScorer(levenshtein.distance))
    expect(knownOptimum(distance)).toBe(0)
    expect(impossibleThreshold(distance, -1)).toBe(true)
  })

  // The helpers being right is not the same as the searches asking them. The
  // failure this guards is someone later reaching past `impossibleThreshold`
  // for `impossibleTrustedThreshold`, which would silently skip a custom
  // scorer's own kernel — the one place its declared bounds get checked.
  test('every search still runs a custom scorer under an impossible threshold', () => {
    let calls = 0
    const scorer = createScorer(
      () => {
        calls++
        return 1
      },
      { direction: 'similarity', bounds: [0, 1], symmetric: true },
    )
    // Each of these finds nothing, which is the right answer — a score of 1
    // does not clear a threshold of 2. What is being asserted is that the
    // scorer was asked, rather than skipped on the strength of bounds it merely
    // claimed. A trusted scorer with these bounds would answer without running.
    // Both choices are scored every time: with no known optimum, no scan may
    // stop early either.
    const run = (query: () => unknown, empty: unknown): void => {
      calls = 0
      expect(query()).toEqual(empty)
      expect(calls).toBe(2)
    }
    const matcher = createMatcher(['a', 'b'], { scorer })
    run(() => matcher.best('x', { threshold: 2 }), undefined)
    run(() => matcher.search('x', { threshold: 2 }), [])
    run(() => [...matcher.searchIter('x', { threshold: 2 })], [])
    run(() => bestMatch('x', ['a', 'b'], { scorer, threshold: 2 }), undefined)
    run(() => search('x', ['a', 'b'], { scorer, threshold: 2, limit: 2 }), [])
    run(() => [...searchIter('x', ['a', 'b'], { scorer, threshold: 2 })], [])
  })
})
