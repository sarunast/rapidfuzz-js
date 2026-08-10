// Not ported from RapidFuzz — upstream has no equivalent, because in Python
// this is not a hazard.
//
// Python iterates `str` by code point, JavaScript by UTF-16 code unit:
// `len("😀") == 1` but `"😀".length === 2`. Every scorer here goes through
// `toCodePoints` in `src/_common.ts` so element counts match Python's. Without
// it, astral characters would count double and silently change every score.
//
// The expected values below were produced by upstream's pure-Python backend
// (`rapidfuzz.distance.Levenshtein_py` etc.), not by this implementation.
import { describe, expect, it } from 'vitest'

import { indelDistance } from '../src/distance/indel.js'
import { levenshteinDistance, levenshteinEditops } from '../src/distance/levenshtein.js'
import {
  prefixDistance,
  prefixNormalizedSimilarity,
  prefixSimilarity,
} from '../src/distance/prefix.js'
import { ratio } from '../src/fuzz.js'
import { defaultProcess } from '../src/utils.js'
import { editopTuples, maxLen } from './common.js'

interface UnicodeCase {
  readonly name: string
  readonly s1: string
  readonly s2: string
  readonly levenshtein: number
  readonly indel: number
  readonly ratio: number
  readonly editops: ReadonlyArray<readonly [string, number, number]>
}

const cases: readonly UnicodeCase[] = [
  {
    name: 'astral characters count as one element',
    s1: '\u{1F600}',
    s2: '\u{1F601}',
    levenshtein: 1,
    indel: 2,
    ratio: 0,
    editops: [['replace', 0, 0]],
  },
  {
    name: 'an astral character is a single insertion',
    s1: 'a\u{1F600}b',
    s2: 'ab',
    levenshtein: 1,
    indel: 1,
    ratio: 80,
    editops: [['delete', 1, 1]],
  },
  {
    // A ZWJ sequence is five code points, so it is five elements — upstream is
    // not grapheme-aware either, and matching that is the point.
    name: 'a ZWJ emoji sequence is not one grapheme',
    s1: '\u{1F468}‍\u{1F469}‍\u{1F467}',
    s2: '\u{1F468}',
    levenshtein: 4,
    indel: 4,
    ratio: 33.333333333333336,
    editops: [
      ['delete', 1, 1],
      ['delete', 2, 1],
      ['delete', 3, 1],
      ['delete', 4, 1],
    ],
  },
  {
    name: 'a flag is its two regional indicators',
    s1: '\u{1F1E8}\u{1F1ED}',
    s2: '\u{1F1E9}\u{1F1EA}',
    levenshtein: 2,
    indel: 4,
    ratio: 0,
    editops: [
      ['replace', 0, 0],
      ['replace', 1, 1],
    ],
  },
  {
    // No NFC normalization anywhere, matching upstream.
    name: 'a combining mark is not folded into its base character',
    s1: 'é',
    s2: 'é',
    levenshtein: 2,
    indel: 3,
    ratio: 0,
    editops: [
      ['replace', 0, 0],
      ['delete', 1, 1],
    ],
  },
  {
    name: 'a skin-tone modifier is its own element',
    s1: '\u{1F44D}\u{1F3FD}',
    s2: '\u{1F44D}',
    levenshtein: 1,
    indel: 1,
    ratio: 66.66666666666667,
    editops: [['delete', 1, 1]],
  },
  {
    name: 'positions are code-point offsets, not UTF-16 offsets',
    s1: 'café \u{1F600}',
    s2: 'cafe \u{1F600}',
    levenshtein: 1,
    indel: 2,
    ratio: 83.33333333333334,
    editops: [['replace', 3, 3]],
  },
]

describe.each(cases)('$name', (c) => {
  it('matches upstream levenshteinDistance', () => {
    expect(levenshteinDistance(c.s1, c.s2)).toBe(c.levenshtein)
  })

  it('matches upstream indelDistance', () => {
    expect(indelDistance(c.s1, c.s2)).toBe(c.indel)
  })

  it('matches upstream ratio', () => {
    expect(ratio(c.s1, c.s2)).toBe(c.ratio)
  })

  it('matches upstream levenshteinEditops', () => {
    expect(editopTuples(levenshteinEditops(c.s1, c.s2))).toEqual(
      c.editops.map((op) => [...op]),
    )
  })
})

it('splitting a string by code point compares equal to the string', () => {
  const emoji = 'a\u{1F600}b'
  expect(levenshteinDistance(emoji, [...emoji])).toBe(0)
  expect(ratio(emoji, [...emoji])).toBe(100)
})

// `apply` reconstructs the destination by walking positions that count code
// points, and it takes a fast path for text UTF-16 can index directly. Every
// other test of it is ASCII, which is to say every other test of it exercises
// only that fast path. These are the ones that would catch a position read in
// the wrong units — and they cover a mixed pair as well, because the two
// arguments are specialised separately and the wrong choice on one side is a
// silent half-character shift rather than an error.
describe('applying an edit script to astral text', () => {
  const PAIRS: ReadonlyArray<readonly [string, string, string]> = [
    ['astral on both sides', 'a\u{1F600}bc\u{1F44D}', 'a\u{1F601}xc\u{1F44D}z'],
    ['astral source, BMP destination', '\u{1F600}\u{1F601}abc', 'xabcd'],
    ['BMP source, astral destination', 'xabcd', '\u{1F600}\u{1F601}abc'],
    ['every character astral', '\u{1F600}\u{1F601}\u{1F602}', '\u{1F602}\u{1F600}'],
    ['an astral character alone', '\u{1F600}', ''],
    ['inserted into empty', '', '\u{1F600}\u{1F44D}'],
  ]

  for (const [name, s1, s2] of PAIRS) {
    it(name, () => {
      const ops = levenshteinEditops(s1, s2)

      expect(ops.apply(s1, s2)).toBe(s2)
      expect(ops.toOpcodes().apply(s1, s2)).toBe(s2)
      expect(ops.inverse().apply(s2, s1)).toBe(s1)
    })
  }

  it('reads a lone surrogate as one element, as Python does', () => {
    // Unpaired, so UTF-16 indexing splits nothing and the fast path is correct.
    const s1 = 'a\uD800b'
    const s2 = 'a\uD800c'
    const ops = levenshteinEditops(s1, s2)

    expect(ops.apply(s1, s2)).toBe(s2)
    expect(ops.toOpcodes().apply(s1, s2)).toBe(s2)
  })
})

it('treats emoji as non-word characters in defaultProcess', () => {
  expect(defaultProcess('Hello \u{1F600} World! \u{1F44D}\u{1F3FD}')).toBe(
    'hello   world',
  )
})

// The generic suite computes every expected normalised score from `maxLen`, so
// a `maxLen` that counted UTF-16 units would expect the wrong value for astral
// input rather than fail against the scorer — a trap that only springs once one
// of these inputs reaches the generic wrapper.
describe('the suite measures length the way a scorer does', () => {
  it('counts an astral character once', () => {
    expect(maxLen('\u{1F600}a', '\u{1F600}b')).toBe(2)
    expect(maxLen('\u{1F600}', '\u{1F600}')).toBe(1)
    expect(maxLen('abc', 'ab')).toBe(3)
    expect(maxLen([1, 2, 3], Uint8Array.of(1, 2))).toBe(3)
  })

  it('agrees with what the scorers themselves report', () => {
    expect(prefixSimilarity('\u{1F600}', '\u{1F600}')).toBe(1)
    expect(prefixDistance('\u{1F600}a', '\u{1F600}b')).toBe(1)
    expect(prefixNormalizedSimilarity('\u{1F600}a', '\u{1F600}b')).toBe(0.5)
    expect(prefixNormalizedSimilarity('\u{1F600}a', '\u{1F600}b')).toBe(
      prefixSimilarity('\u{1F600}a', '\u{1F600}b') / maxLen('\u{1F600}a', '\u{1F600}b'),
    )
  })
})
