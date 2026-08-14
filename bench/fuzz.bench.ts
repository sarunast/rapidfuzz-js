import { fuzzPartialRatio } from '../src/fuzz/partialRatio.js'
import { fuzzRatio } from '../src/fuzz/ratio.js'
import { fuzzPartialTokenSetRatio } from '../src/fuzz/token/partialTokenSetRatio.js'
import { fuzzTokenSetRatio } from '../src/fuzz/token/tokenSetRatio.js'
import { fuzzTokenSortRatio } from '../src/fuzz/token/tokenSortRatio.js'
import { fuzzWeightedRatio } from '../src/fuzz/weightedRatio.js'
import { pairs, sentences, similarPairs } from './tooling/corpus.js'
import { describe, measure } from './tooling/harness.js'

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
// is already a millisecond. See `tooling/harness.ts` for why a sample is kept there.
const partialHaystacksFew = partialHaystacks.slice(0, 8)
const partialHaystacksSome = partialHaystacks.slice(0, 16)

// Haystacks that actually contain the needle, which none of the above do:
// `partialNeedle` is repeated alphabet and the corpus is random sentences, so
// every case built from the two scores badly everywhere and the bisection
// improves on its running best only a handful of times. That is the shape a
// scan is *cheapest* on, and the opposite of what `extract` sees — there the
// haystack holds a near-match, the running best keeps improving, and every
// improvement is an endpoint the search has to do something about. Composed
// here rather than in `tooling/corpus.ts`, which is hashed into all 155 baseline
// entries; this file is hashed into its own.
const plantedHaystacks = partialHaystacksFew.map((haystack) => {
  const at = Math.floor(haystack.length / 3)
  return haystack.slice(0, at) + partialNeedle + haystack.slice(at + partialNeedle.length)
})
// The same, with every fourth element of the planted copy replaced, so the best
// window is a near-match rather than an exact one and no scan can stop early.
const nearHaystacks = partialHaystacksFew.map((haystack) => {
  const at = Math.floor(haystack.length / 3)
  const near = Array.from(partialNeedle, (c, i) => (i % 4 === 0 ? 'z' : c)).join('')
  return haystack.slice(0, at) + near + haystack.slice(at + near.length)
})

// Each case inlines its own loop rather than sharing a `run(data, fn)` helper —
// V8's inline caches live on the function literal, so one helper would leave
// every scorer here sharing a megamorphic call site.

describe('fuzzRatio', () => {
  measure('8 chars', () => {
    for (const [a, b] of shortPairs) fuzzRatio(a, b)
  })
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) fuzzRatio(a, b)
  })
  measure('128 chars', () => {
    for (const [a, b] of longPairs) fuzzRatio(a, b)
  })
  measure('128 chars, BMP', () => {
    for (const [a, b] of bmpPairs) fuzzRatio(a, b)
  })
  measure('128 chars, astral', () => {
    for (const [a, b] of astralPairs) fuzzRatio(a, b)
  })
})

describe('partialRatio', () => {
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) fuzzPartialRatio(a, b)
  })
  measure('128 chars', () => {
    for (const [a, b] of longPairs) fuzzPartialRatio(a, b)
  })
  measure('sentences', () => {
    for (const [a, b] of sentencePairs) fuzzPartialRatio(a, b)
  })
  measure('128 chars in long haystack', () => {
    for (const haystack of partialHaystacksFew) fuzzPartialRatio(partialNeedle, haystack)
  })
  measure('128 chars in long haystack, cutoff 90', () => {
    for (const haystack of partialHaystacksFew) {
      fuzzPartialRatio(partialNeedle, haystack, { scoreCutoff: 90 })
    }
  })
  measure('128 chars planted in long haystack', () => {
    for (const haystack of plantedHaystacks) fuzzPartialRatio(partialNeedle, haystack)
  })
  measure('128 chars near-matched in long haystack', () => {
    for (const haystack of nearHaystacks) fuzzPartialRatio(partialNeedle, haystack)
  })
})

describe('token scorers', () => {
  measure('tokenSortRatio, sentences', () => {
    for (const [a, b] of sentencePairs) fuzzTokenSortRatio(a, b)
  })
  measure('tokenSetRatio, sentences', () => {
    for (const [a, b] of sentencePairs) fuzzTokenSetRatio(a, b)
  })
  measure('partialTokenSetRatio, sentences', () => {
    for (const [a, b] of sentencePairs) fuzzPartialTokenSetRatio(a, b)
  })
})

// Adaptive fuzzy similarity is a common search scorer, so it is the single hottest
// path in the library for anyone using it as intended.
describe('weightedRatio', () => {
  measure('32 chars', () => {
    for (const [a, b] of mediumPairs) fuzzWeightedRatio(a, b)
  })
  measure('sentences', () => {
    for (const [a, b] of sentencePairs) fuzzWeightedRatio(a, b)
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
    for (let i = 0; i < 200; i++) fuzzPartialRatio(needle, shortHaystack)
  })
  measure('16 interior windows, cutoff 90', () => {
    for (let i = 0; i < 200; i++) {
      fuzzPartialRatio(needle, shortHaystack, { scoreCutoff: 90 })
    }
  })
  measure('many interior windows', () => {
    for (let i = 0; i < 32; i++) fuzzPartialRatio(needle, longHaystack)
  })
  // The bisection's book-keeping is sized on the haystack while the search
  // reads a handful of windows, so a long haystack whose first window is a
  // perfect match is where that gap is widest — the scan stops immediately and
  // everything else it prepared was waste. Composed here from `partialNeedle`
  // rather than added to `tooling/corpus.ts`, which is hashed into all 155 baseline
  // entries where this file is hashed only into its own.
  const plantedAtStart = needle + 'z'.repeat(8_000)

  measure('perfect first window in an 8k haystack', () => {
    for (let i = 0; i < 32; i++) fuzzPartialRatio(needle, plantedAtStart)
  })
})

// A 33-64 element pattern spans exactly two words, the width a specialised
// kernel would target. Nothing else here sits there: 32 chars is one word and
// 128 is four.
describe('partialRatio, two-word needle', () => {
  const needle = partialNeedle.slice(0, 48)

  measure('48-char needle in long haystacks', () => {
    for (const haystack of partialHaystacksSome) fuzzPartialRatio(needle, haystack)
  })
})

// The other width the row vector used to serve: 65 to 96 elements is three
// words, and the case above is the only other one in the suite that is neither
// one word nor four.
describe('partialRatio, three-word needle', () => {
  const needle = partialNeedle.slice(0, 80)

  measure('80-char needle in long haystacks', () => {
    for (const haystack of partialHaystacksSome) fuzzPartialRatio(needle, haystack)
  })
})
