import { describe } from 'vitest'

import {
  partialRatio,
  partialTokenSetRatio,
  qRatio,
  ratio,
  tokenSetRatio,
  tokenSortRatio,
  wRatio,
} from '../src/fuzz.js'
import { pairs, sentences, similarPairs } from './_corpus.js'
import { measure } from './_harness.js'

const shortPairs = similarPairs(200, 8)
const mediumPairs = similarPairs(200, 32)
const longPairs = similarPairs(100, 128)
const sentencePairs = pairs(sentences(200, 6))
const unicodeSource = similarPairs(100, 128, 0xb0b0_b0b0)

/**
 * `similarPairs` draws from `a`-`z`, so shifting its output into Cyrillic is
 * what actually reaches the shared table's growth path past Latin-1. Without
 * the shift this case measured ASCII a second time under a name that said
 * otherwise.
 */
function toCyrillic(value: string): string {
  return Array.from(value, (ch) => String.fromCharCode(ch.charCodeAt(0) + 0x03d0)).join(
    '',
  )
}

const bmpPairs = unicodeSource.map(([a, b]): readonly [string, string] => [
  toCyrillic(a),
  toCyrillic(b),
])
const astralPairs = unicodeSource.map(([a, b]): readonly [string, string] => [
  a.replaceAll('a', '😀'),
  b.replaceAll('a', '😀'),
])
const partialNeedle = 'abcdefghijklmnopqrstuvwxyz'.repeat(5).slice(0, 128)
const partialHaystacks = sentences(50, 80, 0x71a1_5eed)

// `partialRatio` scans O(n) windows per call, so a handful of these haystacks
// is already a millisecond. See `_harness.ts` for why a sample is kept there.
const partialHaystacksFew = partialHaystacks.slice(0, 8)
const partialHaystacksSome = partialHaystacks.slice(0, 16)

// Each case inlines its own loop rather than sharing a `run(data, fn)` helper —
// V8's inline caches live on the function literal, so one helper would leave
// every scorer here sharing a megamorphic call site.

describe('ratio', () => {
  measure('8 chars', () => {
    for (const [a, b] of shortPairs) ratio(a, b)
  })
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) ratio(a, b)
  })
  measure('128 chars', () => {
    for (const [a, b] of longPairs) ratio(a, b)
  })
  measure('128 chars, BMP', () => {
    for (const [a, b] of bmpPairs) ratio(a, b)
  })
  measure('128 chars, astral', () => {
    for (const [a, b] of astralPairs) ratio(a, b)
  })
})

describe('partialRatio', () => {
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) partialRatio(a, b)
  })
  measure('128 chars', () => {
    for (const [a, b] of longPairs) partialRatio(a, b)
  })
  measure('sentences', () => {
    for (const [a, b] of sentencePairs) partialRatio(a, b)
  })
  measure('128 chars in long haystack', () => {
    for (const haystack of partialHaystacksFew) partialRatio(partialNeedle, haystack)
  })
  measure('128 chars in long haystack, cutoff 90', () => {
    for (const haystack of partialHaystacksFew) {
      partialRatio(partialNeedle, haystack, { scoreCutoff: 90 })
    }
  })
})

describe('token scorers', () => {
  measure('tokenSortRatio, sentences', () => {
    for (const [a, b] of sentencePairs) tokenSortRatio(a, b)
  })
  measure('tokenSetRatio, sentences', () => {
    for (const [a, b] of sentencePairs) tokenSetRatio(a, b)
  })
  measure('partialTokenSetRatio, sentences', () => {
    for (const [a, b] of sentencePairs) partialTokenSetRatio(a, b)
  })
})

// WRatio is what `process.extract` defaults to, so it is the single hottest
// path in the library for anyone using it as intended.
describe('wRatio', () => {
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) wRatio(a, b)
  })
  measure('sentences', () => {
    for (const [a, b] of sentencePairs) wRatio(a, b)
  })
})

describe('qRatio', () => {
  measure('sentences', () => {
    for (const [a, b] of sentencePairs) qRatio(a, b)
  })
})

// The full-length window scan switches between a linear sweep and the
// divide-and-conquer search at 64 interior windows, and nothing else covers
// either side of that boundary.
describe('partialRatio window scan', () => {
  const needle = partialNeedle.slice(0, 64)
  const shortHaystack = needle + 'qwertyuiopasdfgh'
  const longHaystack = partialHaystacks[0] ?? ''

  measure('16 interior windows', () => {
    for (let i = 0; i < 200; i++) partialRatio(needle, shortHaystack)
  })
  measure('16 interior windows, cutoff 90', () => {
    for (let i = 0; i < 200; i++) {
      partialRatio(needle, shortHaystack, { scoreCutoff: 90 })
    }
  })
  measure('many interior windows', () => {
    for (let i = 0; i < 32; i++) partialRatio(needle, longHaystack)
  })
})

// A 33-64 element pattern spans exactly two words, the width a specialised
// kernel would target. Nothing else here sits there: 32 chars is one word and
// 128 is four.
describe('partialRatio, two-word needle', () => {
  const needle = partialNeedle.slice(0, 48)

  measure('48-char needle in long haystacks', () => {
    for (const haystack of partialHaystacksSome) partialRatio(needle, haystack)
  })
})
