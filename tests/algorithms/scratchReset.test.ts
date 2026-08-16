// Not ported from RapidFuzz — upstream has no equivalent, because CPython does
// not retain scratch across calls the way these kernels do.
//
// The kernels reuse module-level buffers that grow on demand and never shrink,
// and the shared bitmask builder additionally keeps a symbol table that widens
// permanently, a one-entry mask memo and a generation counter. The benchmarks
// clear all of that between cases so that a case's number does not depend on
// which case ran before it — see `bench/harness/harness.ts`.
//
// That makes the reset functions load-bearing for the benchmark methodology in
// a way nothing else checks: a buffer they forget leaves the order dependence
// they exist to remove, and a value they clear that a later call needed would
// be a correctness bug. Both directions are covered here.
//
// The workload is deliberately wide: text that widens the shared table past
// Latin-1, astral text that takes the `Map` instead, a repeated pattern that
// fills the mask memo, non-uniform weights for the generic dynamic program,
// long inputs for the blocked kernels, and Jaro and Damerau, which keep scratch
// of their own.
import { describe, expect, it } from 'vitest'

import {
  maskPoolOf,
  resetBitVectorScratch,
} from '../../src/algorithms/bitmask/blockMasks.js'
import {
  damerauLevenshteinDistance,
  resetDamerauScratch,
} from '../../src/algorithms/damerauLevenshtein/implementation.js'
import { indelDistance } from '../../src/algorithms/indel/implementation.js'
import {
  jaroSimilarity,
  resetJaroScratch,
} from '../../src/algorithms/jaro/implementation.js'
import {
  lcsSeqEditops,
  lcsSeqNormalizedSimilarity,
} from '../../src/algorithms/lcs/implementation.js'
import { levenshteinEditops } from '../../src/algorithms/levenshtein/editops.js'
import { resetWeightedScratch } from '../../src/algorithms/levenshtein/internal/scratch.js'
import { levenshteinDistance } from '../../src/algorithms/levenshtein/metric.js'
import { osaDistance } from '../../src/algorithms/osa/implementation.js'
import {
  osaRetainedBytes,
  resetOsaScratch,
} from '../../src/algorithms/osa/internal/kernel.js'
import { matrixScores } from '../../testing/scoreMatrix.js'

function resetAll(): void {
  resetBitVectorScratch()
  resetOsaScratch()
  resetWeightedScratch()
  resetJaroScratch()
  resetDamerauScratch()
}

const ascii = ['kitten', 'sitting', 'abcdefghijklmnop', 'abcdefghijklmnoq']
const cyrillic = ['привет мир', 'привет мор']
const astral = ['🦊abc🚀', '🦊abd🌍']
const long = ['a'.repeat(300) + 'kitten', 'a'.repeat(300) + 'sitting']
const wide = ['éàüñçÿ'.repeat(20), 'éàüñçy'.repeat(20)]

/**
 * Every distinct shape of retained state, run as one sequence so that later
 * entries see whatever the earlier ones left behind.
 */
function workload(): unknown[] {
  const results: unknown[] = []
  const pairs = [ascii, cyrillic, astral, long, wide]

  for (const [a, b] of pairs) {
    results.push(
      levenshteinDistance(a, b),
      levenshteinDistance(a, b, { weights: [3, 7, 5] }),
      levenshteinDistance(a, b, { weights: [1.5, 2.25, 3.5] }),
      levenshteinDistance(a, b, { scoreCutoff: 4, scoreHint: 2 }),
      indelDistance(a, b),
      lcsSeqNormalizedSimilarity(a, b),
      osaDistance(a, b),
      damerauLevenshteinDistance(a, b),
      jaroSimilarity(a, b),
      levenshteinEditops(a, b).operations.length,
      lcsSeqEditops(a, b).operations.length,
    )
  }

  // The prepared paths keep query-side state of their own, and are what the
  // shared mask memo exists for.
  results.push(
    matrixScores(['привет', 'kitten'], ['привет мир', 'sitting'], {
      scorer: levenshteinDistance,
    }),
    matrixScores(astral, astral, { scorer: jaroSimilarity }),
  )
  return results
}

describe('benchmark scratch resets', () => {
  it('does not change any result', () => {
    const withoutReset = workload()

    resetAll()
    const afterReset = workload()

    expect(afterReset).toStrictEqual(withoutReset)
  })

  it('is idempotent, and safe on state that was never built', () => {
    resetAll()
    resetAll()

    expect(workload()).toStrictEqual((resetAll(), workload()))
  })

  it('leaves no state that changes a later answer', () => {
    // Reset between every step rather than only at the start: if any retained
    // value were load-bearing across calls, dropping it mid-sequence would
    // show up here and nowhere else.
    const contiguous = workload()

    resetAll()
    const interrupted: unknown[] = []
    for (const value of workload()) {
      interrupted.push(value)
      resetAll()
    }

    expect(interrupted).toStrictEqual(contiguous)
  })

  it('starts each run from the same state, whatever ran before it', () => {
    // The order dependence the resets exist to remove: a widened symbol table
    // and grown buffers are permanent, so without a reset the second reading
    // would come from a differently-sized table than the first.
    resetAll()
    const cold = workload()

    lcsSeqEditops('a'.repeat(4096), 'b'.repeat(4096))
    levenshteinDistance('一'.repeat(500), '丁'.repeat(500))
    resetAll()

    expect(workload()).toStrictEqual(cold)
  })
})

// The reset tests above compare answers, which cannot see a buffer that is
// still allocated. OSA's scratch is reached through eight cached `subarray`
// views, and a view keeps the whole backing buffer alive — so replacing the
// scratch binding alone frees nothing until the next call rebuilds the views.
describe('the OSA scratch after a reset', () => {
  const pattern = Array.from({ length: 4096 }, (_, i) => 97 + (i % 5))
  // Differing at both ends, so the affix trim leaves the full width for the
  // blocked kernel, which is the only path that allocates the scratch.
  const text = [90, ...pattern.slice(1, -1), 91]

  it('holds a buffer while it is in use', () => {
    resetOsaScratch()
    osaDistance(pattern, text)

    expect(osaRetainedBytes()).toBeGreaterThan(1024)
  })

  it('holds nothing afterwards, views included', () => {
    osaDistance(pattern, text)
    resetOsaScratch()

    expect(osaRetainedBytes()).toBe(0)
  })

  it('still answers the same once the views have been dropped', () => {
    resetOsaScratch()
    const cold = osaDistance(pattern, text)

    resetOsaScratch()
    expect(osaDistance(pattern, text)).toBe(cold)
    expect(cold).toBe(2)
  })
})

// Every other buffer here is linear in the input or bounded by the symbol space.
// The mask pool is neither: it holds a block of `words` per distinct element, so
// a sequence that repeats nothing takes `length * words` — 48.8 MB for two
// 20,000-element arrays of distinct objects, which the API accepts because an
// element is whatever the caller's sequence holds. That much is unavoidable
// during the comparison; keeping it afterwards is not, and the comparison is
// the last thing a process runs often enough for "the next build clears it" to
// be no answer at all.
//
// Which owner a build's masks go to is covered where the builders are, in
// `src/algorithms/bitmask/blockMasks.test.ts`. What is only visible from here
// is that a public call leaves nothing oversized behind.
describe('the mask pool past its retention cap', () => {
  const RETAINED_MASK_WORDS = 1 << 19

  /** Distinct objects, so every element takes a block of its own. */
  function unique(count: number, tag: string): ReadonlyArray<unknown> {
    return Array.from({ length: count }, (_, index) => ({ tag, index }))
  }

  // One element past the largest pattern the memo takes, which is the first
  // size whose masks the module refuses to keep.
  const wideA = unique(4097, 'a')
  const wideB = unique(4097, 'b')

  it('is back within the cap once the comparison has returned', () => {
    resetAll()

    expect(levenshteinDistance(wideA, wideB)).toBe(4097)
    expect(maskPoolOf().length).toBeLessThanOrEqual(RETAINED_MASK_WORDS)
  })

  // Both sides of the handover. A build climbing past the cap leaves the module
  // holding exactly the cap — the sizes on the way up are still worth keeping,
  // and the one that cleared it is not — and the follow-up then reuses that
  // buffer rather than growing one of its own.
  it('stops at the cap, and leaves that much to reuse', () => {
    resetAll()
    levenshteinDistance(wideA, wideB)
    expect(maskPoolOf().length).toBe(RETAINED_MASK_WORDS)

    const held = maskPoolOf()
    const repetitive = 'a'.repeat(3000)
    expect(levenshteinDistance(repetitive, `b${repetitive.slice(1, -1)}c`)).toBe(2)
    expect(maskPoolOf()).toBe(held)

    // Nothing shrinks it back either: bounded scratch is the part that stays.
    expect(levenshteinDistance('kitten', 'sitting')).toBe(3)
    expect(maskPoolOf()).toBe(held)
  })

  // `length * words` is the worst case, not the need: past 4096 elements a
  // sequence clears the cap on that bound alone, so a follow-up of one repeated
  // character — 157 words of masks — used to vote for keeping an 8 MB pool.
  it('cannot be held open by a long low-cardinality follow-up', () => {
    resetAll()
    levenshteinDistance(wideA, wideB)

    const repetitive = 'a'.repeat(5000)
    expect(levenshteinDistance(repetitive, `b${repetitive.slice(1, -1)}c`)).toBe(2)
    expect(maskPoolOf().length).toBeLessThanOrEqual(RETAINED_MASK_WORDS)
  })

  it('answers the same either side of the cap', () => {
    resetAll()
    const cold = levenshteinDistance(wideA, wideB)

    levenshteinDistance('kitten', 'sitting')
    expect(levenshteinDistance(wideA, wideB)).toBe(cold)
  })

  it('answers the same for a string pattern either side of the cap', () => {
    resetAll()
    const long = Array.from({ length: 4096 }, (_, i) => String.fromCharCode(0x4e00 + i))
    const pattern = long.join('')
    const text = [...long].reverse().join('')

    const cold = levenshteinDistance(pattern, text)
    expect(levenshteinDistance(pattern, text)).toBe(cold)
    expect(maskPoolOf().length).toBeLessThanOrEqual(RETAINED_MASK_WORDS)
  })
})

// The shared symbol table starts at Latin-1 and widens permanently, so which
// builder does the widening depends on what ran before it. After a reset the
// first comparison is the one that widens, and a multi-word pattern reaches a
// different builder from a single-word one.
describe('widening the shared table from each mask builder', () => {
  const cyrillic = 'привет мир как дела сегодня вечером друзья мои'
  // Scattered rather than one edit: every kernel trims the common affix first,
  // so a pair differing in one place is a one-element pair by the time a mask
  // builder sees it — and the multi-word builder is the point of this.
  const edited = [...cyrillic].map((c, i) => (i % 7 === 0 ? 'Ж' : c)).join('')

  it('widens from a multi-word pattern', () => {
    resetAll()

    expect(cyrillic.length).toBeGreaterThan(32)
    expect(levenshteinDistance(cyrillic, edited)).toBeGreaterThan(4)
    expect(levenshteinDistance(cyrillic, edited)).toBe(
      levenshteinDistance(edited, cyrillic),
    )
    expect(lcsSeqNormalizedSimilarity(cyrillic, edited)).toBeGreaterThan(0)
    expect(indelDistance(cyrillic, edited)).toBeGreaterThan(0)
  })

  // The multi-word builder keeps a loop of its own for a pattern that is not a
  // string, and that loop widens the table separately.
  it('widens from a multi-word pattern of elements', () => {
    resetAll()

    const source = [...cyrillic].map((c) => c.codePointAt(0))
    const destination = [...edited].map((c) => c.codePointAt(0))

    expect(levenshteinDistance(source, destination)).toBe(
      levenshteinDistance(cyrillic, edited),
    )
    expect(lcsSeqNormalizedSimilarity(source, destination)).toBe(
      lcsSeqNormalizedSimilarity(cyrillic, edited),
    )
  })

  it('widens from a single-word pattern', () => {
    resetAll()

    const short = cyrillic.slice(0, 20)
    const shortEdited = edited.slice(0, 20)
    expect(levenshteinDistance(short, shortEdited)).toBeGreaterThan(1)
    expect(lcsSeqNormalizedSimilarity(short, shortEdited)).toBeGreaterThan(0)
  })
})
