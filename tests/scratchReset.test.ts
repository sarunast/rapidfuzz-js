// Not ported from RapidFuzz — upstream has no equivalent, because CPython does
// not retain scratch across calls the way these kernels do.
//
// The kernels reuse module-level buffers that grow on demand and never shrink,
// and `_bitVector/shared.ts` additionally keeps a shared symbol table that widens
// permanently, a one-entry mask memo and a generation counter. The benchmarks
// clear all of that between cases so that a case's number does not depend on
// which case ran before it — see `bench/_harness.ts`.
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

import { resetSharedScratch } from '../src/distance/_bitVector/index.js'
import {
  damerauLevenshteinDistance,
  resetDamerauScratch,
} from '../src/distance/damerauLevenshtein.js'
import { indelDistance } from '../src/distance/indel.js'
import { jaroSimilarity, resetJaroScratch } from '../src/distance/jaro.js'
import { lcsSeqEditops, lcsSeqSimilarity } from '../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinEditops,
  resetWeightedScratch,
} from '../src/distance/levenshtein.js'
import { osaDistance } from '../src/distance/osa.js'
import { extractOne } from '../src/search.js'
import { matrixScores } from './matrix.js'

function resetAll(): void {
  resetSharedScratch()
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
      lcsSeqSimilarity(a, b),
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
    extractOne('kitten', ['sitting', 'kitchen', 'mitten'])?.score,
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
    expect(lcsSeqSimilarity(cyrillic, edited)).toBeGreaterThan(0)
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
    expect(lcsSeqSimilarity(source, destination)).toBe(lcsSeqSimilarity(cyrillic, edited))
  })

  it('widens from a single-word pattern', () => {
    resetAll()

    const short = cyrillic.slice(0, 20)
    const shortEdited = edited.slice(0, 20)
    expect(levenshteinDistance(short, shortEdited)).toBeGreaterThan(1)
    expect(lcsSeqSimilarity(short, shortEdited)).toBeGreaterThan(0)
  })
})
