// Not ported from RapidFuzz — upstream has no counterpart. Its caches are owned
// by a `Cached*` object and die with it; ours are module-level tables reused for
// the life of the process, and a stamp is what tells a live entry from one an
// earlier comparison left behind.
//
// The stamp is an `Int32Array` cell, so the counter that writes it has a
// ceiling. At the ceiling the table has to be cleared and the counter restarted,
// or the next comparison starts matching entries stamped two billion builds ago
// — which reports elements as present in a pattern that does not contain them.
//
// Two billion comparisons is not a test. Both resets therefore take a starting
// generation, which is the only reason they take an argument at all: it puts the
// counter within a few builds of the wrap so the clearing can be driven and its
// effect checked.
import { describe, expect, it } from 'vitest'

import { resetBitVectorScratch } from '../../src/algorithms/bitmask/blockMasks.js'
import {
  damerauLevenshteinDistance,
  resetDamerauScratch,
} from '../../src/algorithms/damerauLevenshtein/implementation.js'
import { indelDistance } from '../../src/algorithms/indel/implementation.js'
import { lcsSeqNormalizedSimilarity } from '../../src/algorithms/lcs/implementation.js'
import { levenshteinDistance } from '../../src/algorithms/levenshtein/metric.js'
import { fuzzPartialRatio } from '../../src/fuzz/partialRatio.js'
import {
  partialRatioAlignment_impl,
  resetPartialRatioScratch,
} from '../../src/fuzz/partialWindow.js'

/** One before the counter's ceiling, so the next build but one wraps it. */
const NEAR_LIMIT = 0x7fff_fffd

// Pairs chosen to span the mask table's three regions — Latin-1, the widened
// high range, and the overflow map — since the wrap clears the stamps of all
// three and a stale entry in any of them scores the same way: too well.
const PAIRS: ReadonlyArray<readonly [string, string, number]> = [
  ['kitten', 'sitting', 3],
  ['abcdefghij', 'abcdefghij', 0],
  ['flaw', 'lawn', 2],
  ['привет мир', 'привет мор', 1],
  ['一二三四五六', '一二三四五七', 1],
  ['a'.repeat(40) + 'b', 'a'.repeat(40) + 'c', 1],
  ['\u{1f600}\u{1f601}ab', '\u{1f600}\u{1f602}ab', 1],
]

describe('the shared mask table across its stamp wrap', () => {
  it('keeps answering exactly while the counter turns over', () => {
    resetBitVectorScratch(NEAR_LIMIT)

    // Several passes, so comparisons land on both sides of the wrap and a
    // pattern built before it is scored again after.
    for (let pass = 0; pass < 4; pass++) {
      for (const [a, b, expected] of PAIRS) {
        expect(levenshteinDistance(a, b), `${a} vs ${b}`).toBe(expected)
      }
    }

    resetBitVectorScratch()
  })

  // The single-word builder stamps the same table through a different path, and
  // the LCS kernels read it rather than Levenshtein's.
  it('keeps the single-word and LCS kernels exact across the wrap', () => {
    resetBitVectorScratch(NEAR_LIMIT)

    for (let pass = 0; pass < 4; pass++) {
      expect(indelDistance('flaw', 'lawn')).toBe(2)
      expect(lcsSeqNormalizedSimilarity('abcde', 'axcye')).toBe(3 / 5)
      expect(indelDistance('привет', 'превет')).toBe(2)
    }

    resetBitVectorScratch()
  })

  // The memo that lets one query serve a list of candidates remembers the
  // generation its masks were built at, and the wrap hands that number back out.
  // So it is dropped there too: a memo kept across it would answer from a table
  // the wrap had just emptied.
  it('keeps the multi-word mask memo exact across the wrap', () => {
    resetBitVectorScratch(NEAR_LIMIT)

    const query = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEF'
    const candidates = [query, 'FEDCBA9876543210zyxwvutsrqponmlkjihgfedcba']
    const cold = candidates.map((c) => levenshteinDistance(query, c))

    for (let pass = 0; pass < 4; pass++) {
      expect(candidates.map((c) => levenshteinDistance(query, c))).toEqual(cold)
    }

    resetBitVectorScratch()
    expect(cold).toEqual(candidates.map((c) => levenshteinDistance(query, c)))
  })

  // A stale stamp would report a match for an element the new pattern does not
  // hold, so the pair that catches it is one sharing nothing with the pattern
  // built just before the wrap.
  it('does not carry a match across the wrap', () => {
    resetBitVectorScratch(NEAR_LIMIT)

    expect(levenshteinDistance('abcdef', 'abcdef')).toBe(0)
    expect(levenshteinDistance('ghijkl', 'mnopqr')).toBe(6)
    expect(levenshteinDistance('abcdef', 'mnopqr')).toBe(6)

    resetBitVectorScratch()
  })
})

// `partialRatio`'s window bisection keeps the same kind of table for a different
// question — "has this window been scored yet" rather than "does the pattern
// hold this element" — and a stale stamp there is read as a distance the
// current haystack never produced.
describe('the window bisection across its stamp wrap', () => {
  const needle = 'abcdefghijklmnopqrstuvwxyz'.repeat(3).slice(0, 64)
  // Padding chosen twice over: past 64 interior windows so the bisection runs
  // at all rather than the linear scan, and past 256 so the held buffers have
  // to grow rather than being born big enough.
  const planted = 'q'.repeat(300) + needle + 'z'.repeat(300)
  const nearby = `${'q'.repeat(300) + needle.slice(0, 40)}xxxx${needle.slice(44)}${'z'.repeat(300)}`

  it('keeps answering exactly while the counter turns over', () => {
    resetPartialRatioScratch(NEAR_LIMIT)

    // Four passes, so comparisons land either side of the wrap and a haystack
    // scored before it is scored again after.
    for (let pass = 0; pass < 4; pass++) {
      expect(fuzzPartialRatio(needle, planted)).toBe(100)
      expect(fuzzPartialRatio(needle, nearby)).toBeCloseTo(93.75, 5)
    }

    resetPartialRatioScratch()
  })

  it('does not carry a scored window across the wrap', () => {
    resetPartialRatioScratch(NEAR_LIMIT)
    const beforeWrap = partialRatioAlignment_impl(needle, nearby)
    // The second bisection is the one that turns the counter over.
    const afterWrap = partialRatioAlignment_impl(needle, nearby)

    resetPartialRatioScratch()
    expect(afterWrap).toEqual(partialRatioAlignment_impl(needle, nearby))
    expect(beforeWrap).toEqual(afterWrap)
  })

  // Above the retention cap the scan allocates its own pair, whose stamps are
  // zeroes rather than a live generation — the same "nothing scored yet" state
  // the held buffers reach by counting.
  it('scores a haystack past the retention cap', () => {
    resetPartialRatioScratch(NEAR_LIMIT)
    const huge = needle + 'z'.repeat(70_000)

    expect(fuzzPartialRatio(needle, huge)).toBe(100)

    resetPartialRatioScratch()
  })
})

// The seam is only as safe as the value it is given. A generation below 1 puts
// the first build's stamp at 0, which is what a slot no build has touched
// already holds — the table then answers for every element at once, and
// `levenshteinDistance` over two strings sharing nothing returns 0.
describe('the starting generation the reset hooks accept', () => {
  const hooks: ReadonlyArray<readonly [string, (start?: number) => void]> = [
    ['resetBitVectorScratch', resetBitVectorScratch],
    ['resetDamerauScratch', resetDamerauScratch],
    ['resetPartialRatioScratch', resetPartialRatioScratch],
  ]

  it.each(hooks)('%s refuses a generation no stamp can hold', (_name, reset) => {
    for (const bad of [-1, 1.5, NaN, Infinity, 0x7fff_ffff]) {
      expect(() => reset(bad)).toThrow(RangeError)
    }

    expect(() => reset(0)).not.toThrow()
    expect(() => reset(NEAR_LIMIT)).not.toThrow()
    reset()
  })
})

describe('the Damerau last-occurrence table across its stamp wrap', () => {
  it('keeps answering exactly while the counter turns over', () => {
    resetDamerauScratch(NEAR_LIMIT)

    for (let pass = 0; pass < 4; pass++) {
      expect(damerauLevenshteinDistance('ca', 'abc')).toBe(2)
      expect(damerauLevenshteinDistance('kitten', 'sitting')).toBe(3)
      expect(damerauLevenshteinDistance('привет', 'преивт')).toBe(2)
      expect(damerauLevenshteinDistance('abcdef', 'abcdef')).toBe(0)
    }

    resetDamerauScratch()
  })

  it('does not carry a last occurrence across the wrap', () => {
    resetDamerauScratch(NEAR_LIMIT)

    expect(damerauLevenshteinDistance('abcdef', 'abcdef')).toBe(0)
    expect(damerauLevenshteinDistance('ghijkl', 'mnopqr')).toBe(6)
    expect(damerauLevenshteinDistance('abcdef', 'mnopqr')).toBe(6)

    resetDamerauScratch()
  })
})
