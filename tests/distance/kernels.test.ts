// Not ported from RapidFuzz — this guards an optimisation of ours, not a
// behaviour of upstream's.
//
// The scorers run on bit-parallel kernels (`src/distance/_bitVector/`) rather
// than the dynamic programs they replaced. The DP was obviously correct; the
// kernels are not, so they are checked against it directly here.
//
// The cases that matter and that the ported suite does not reach:
//
//   - lengths crossing the 32-bit word boundary, where the single-word kernel
//     hands over to the multi-word one and carries have to cross words;
//   - lengths that are exact multiples of 32, where the top word is full;
//   - elements outside ASCII, which miss the fast lookup table and fall back
//     to the Map;
//   - non-numeric elements, which have no bit position at all;
//   - fractional elements, which index no entry of a lookup table even when
//     they fall between two elements that do.
import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import {
  commonAffix,
  lcsSeqMatrix,
  levenshteinMatrix,
  levenshteinMatrixBytes,
  rowBitSet,
  shiftedRowBitSet,
} from '../../src/distance/_bitParallel.js'
import {
  lcsLength,
  lcsLengthPrepared,
  lcsLengthPreparedBounded,
  lcsLengthRange,
  levenshteinUniform,
  osaOneWord,
  preparePattern,
} from '../../src/distance/_bitVector/index.js'
import { damerauLevenshteinDistance } from '../../src/distance/damerauLevenshtein.js'
import { jaroSimilarity } from '../../src/distance/jaro.js'
import { lcsSeqEditops } from '../../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinEditops,
  levenshteinNormalizedDistance,
  levenshteinNormalizedSimilarity,
  levenshteinSimilarity,
  type LevenshteinWeights,
} from '../../src/distance/levenshtein.js'
import { osaDistance } from '../../src/distance/osa.js'
import { partialRatio, partialRatioAlignment, ratio } from '../../src/fuzz.js'
import { editopTuples } from '../common.js'

/** Textbook LCS, O(n*m). Slow and obviously right. */
function lcsReference(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const row = new Uint32Array(s2.length + 1)

  for (let i = 0; i < s1.length; i++) {
    let prevDiag = 0

    for (let j = 0; j < s2.length; j++) {
      const above = row[j + 1]
      row[j + 1] = s1[i] === s2[j] ? prevDiag + 1 : Math.max(above, row[j])
      prevDiag = above
    }
  }

  return row[s2.length]
}

/** Textbook Levenshtein, O(n*m). */
function levenshteinReference(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const row = new Uint32Array(s2.length + 1)
  for (let j = 0; j <= s2.length; j++) row[j] = j

  for (let i = 1; i <= s1.length; i++) {
    let prevDiag = row[0]
    row[0] = i

    for (let j = 1; j <= s2.length; j++) {
      const above = row[j]
      row[j] = Math.min(
        above + 1,
        row[j - 1] + 1,
        prevDiag + (s1[i - 1] === s2[j - 1] ? 0 : 1),
      )
      prevDiag = above
    }
  }

  return row[s2.length]
}

function weightedLevenshteinReference(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  insert: number,
  delete_: number,
  replace: number,
): number {
  const row = new Float64Array(s2.length + 1)
  for (let j = 0; j <= s2.length; j++) row[j] = j * insert
  for (let i = 1; i <= s1.length; i++) {
    let diagonal = row[0]
    row[0] = i * delete_
    for (let j = 1; j <= s2.length; j++) {
      const above = row[j]
      row[j] = Math.min(
        above + delete_,
        row[j - 1] + insert,
        diagonal + (s1[i - 1] === s2[j - 1] ? 0 : replace),
      )
      diagonal = above
    }
  }
  return row[s2.length]
}

/** Textbook OSA, O(n*m) with the two previous rows kept for transpositions. */
function osaReference(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const rows = [0, 1, 2].map(() => new Uint32Array(s2.length + 1))
  let prev2 = rows[0]
  let prev1 = rows[1]
  let curr = rows[2]

  for (let j = 0; j <= s2.length; j++) prev1[j] = j

  for (let i = 1; i <= s1.length; i++) {
    curr[0] = i

    for (let j = 1; j <= s2.length; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1
      let value = Math.min(curr[j - 1] + 1, prev1[j] + 1, prev1[j - 1] + cost)

      if (i > 1 && j > 1 && s1[i - 1] === s2[j - 2] && s1[i - 2] === s2[j - 1]) {
        value = Math.min(value, prev2[j - 2] + 1)
      }

      curr[j] = value
    }

    const spare = prev2
    prev2 = prev1
    prev1 = curr
    curr = spare
  }

  return prev1[s2.length]
}

function codePoints(s: string): number[] {
  return [...s].map((c) => c.codePointAt(0) ?? 0)
}

// A small alphabet makes repeats — and therefore interesting subsequences —
// likely. With a large one nearly every pair has an LCS of almost zero and the
// interesting paths never run.
const smallAlphabet = fc.stringMatching(/^[abc]*$/)
// 'a' below the table limit, the rest above it, including an astral pair.
const wideAlphabet = fc
  .array(fc.constantFrom('a', 'β', '€', '😀', '\u{10FFFF}'))
  .map((chars) => chars.join(''))

describe('bit-parallel kernels agree with the dynamic program', () => {
  it('on small alphabets', () => {
    fc.assert(
      fc.property(smallAlphabet, smallAlphabet, (a, b) => {
        const s1 = codePoints(a)
        const s2 = codePoints(b)
        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
      }),
      { numRuns: 500 },
    )
  })

  it('on non-ASCII elements, which bypass the lookup table', () => {
    fc.assert(
      fc.property(wideAlphabet, wideAlphabet, (a, b) => {
        const s1 = codePoints(a)
        const s2 = codePoints(b)
        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
      }),
      { numRuns: 300 },
    )
  })

  // The direct lookup table starts at Latin-1 and doubles the first time it is
  // asked for something higher, so which side of it an element falls on depends
  // on what has been scored before. A mask build and the kernel that reads
  // those masks have to agree on the width: read an element under a wider limit
  // than it was filed under and the table is consulted for something the `Map`
  // holds, which reports it absent from a pattern it occurs in.
  //
  // The alphabet crosses every doubling, and the lengths reach past one word so
  // the multi-word builder runs too. `\u{10FFFF}` stays above the widest table
  // there is and has to keep working from the `Map`.
  it('on elements spread across every width the table grows through', () => {
    const spread = fc
      .array(fc.constantFrom('a', 'ÿ', 'Ā', 'β', 'ก', '中', '￿', '\u{10FFFF}'), {
        maxLength: 80,
      })
      .map((chars) => chars.join(''))

    fc.assert(
      fc.property(spread, spread, fc.integer({ min: 0, max: 40 }), (a, b, cutoff) => {
        const s1 = codePoints(a)
        const s2 = codePoints(b)
        const exact = levenshteinReference(s1, s2)

        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(exact)
        // Through the scorer rather than the kernel, so an empty input takes
        // the documented sentinel instead of the kernel's unconditional length.
        expect(levenshteinDistance(a, b, { scoreCutoff: cutoff })).toBe(
          exact <= cutoff ? exact : cutoff + 1,
        )
      }),
      { numRuns: 400 },
    )
  })

  // The banded kernel sets up only the words its band opens on, and gives a
  // word its `vp`, `vn` and score entry as the band widens onto it. The row
  // buffers are shared and never shrink, so a word left uninitialised still
  // holds whatever a previous, wider call put there — which is what this
  // reaches for by scoring a 4096-element pair first and then a short one.
  //
  // Lengths start past one word so the multi-word band runs at all, and the
  // cutoffs cross every dispatch boundary the band has: `mbleven` under four,
  // the single-word band to fifteen, and the multi-word band above it.
  it('reads no uninitialised word after a wider call left the buffers dirty', () => {
    const dirty = (): void => {
      const wide = 'q'.repeat(4096)
      levenshteinUniform(wide, `${'q'.repeat(2048)}z${'q'.repeat(2047)}`, 4096)
    }
    const build = (n: number, seed: number, alphabet: string): string => {
      let state = (seed * 2_654_435_761) >>> 0
      let out = ''
      for (let i = 0; i < n; i++) {
        state = (Math.imul(state, 0x0001_9660) + 0x3c6e_f35f) >>> 0
        out += alphabet[(state >>> 8) % alphabet.length]
      }
      return out
    }

    fc.assert(
      fc.property(
        fc.integer({ min: 33, max: 200 }),
        fc.integer({ min: 33, max: 200 }),
        fc.integer({ min: 0, max: 60 }),
        fc.constantFrom('ab', 'abc', 'abcdefgh'),
        (len1, len2, cutoff, alphabet) => {
          const s1 = codePoints(build(len1, len1 * 31 + cutoff, alphabet))
          const s2 = codePoints(build(len2, len2 * 17 + cutoff, alphabet))
          const exact = levenshteinReference(s1, s2)

          dirty()
          const bounded = levenshteinUniform(s1, s2, cutoff)
          // The kernel promises an exact answer at or under the cutoff and only
          // *some* larger value above it — an affix trim that empties one side
          // returns the true distance rather than `cutoff + 1`. The sentinel is
          // the scorer's contract, asserted against `levenshteinDistance` above.
          if (exact <= cutoff) expect(bounded).toBe(exact)
          else expect(bounded).toBeGreaterThan(cutoff)
        },
      ),
      { numRuns: 600 },
    )
  })

  // The widening happens partway through a build, after earlier elements have
  // already been filed. Those all sat below the old limit, so growing the table
  // must leave them where they are — the first element here is one of them, and
  // it only matches itself if its slot survived.
  it('keeps elements filed before a mid-pattern widening', () => {
    const pattern = 'a\u{FFFF}a'
    const text = 'aa'

    expect(levenshteinUniform(codePoints(pattern), codePoints(text))).toBe(1)
    expect(lcsLength(codePoints(pattern), codePoints(text))).toBe(2)
  })

  it('on non-numeric elements, which have no bit position of their own', () => {
    const token = fc.constantFrom({ id: 1 }, { id: 2 }, 'x', 'y', null, true)

    fc.assert(
      fc.property(fc.array(token), fc.array(token), (s1, s2) => {
        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
      }),
      { numRuns: 300 },
    )
  })

  // Regression: elements in `0..255` take a direct index into a typed array,
  // but a typed array has no element at a fractional index — `slots[1.5] = m`
  // was dropped and read back as `undefined`, so `1.5` matched nothing, not
  // even another `1.5`.
  it('on fractional numbers inside the direct-lookup range', () => {
    const token = fc.constantFrom(0.5, 1.5, 1, 2, 127.25, 255.5, 255)

    fc.assert(
      fc.property(fc.array(token), fc.array(token), (s1, s2) => {
        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
      }),
      { numRuns: 300 },
    )
  })

  it('with enough distinct fractional numbers to need several words', () => {
    const s1 = Array.from({ length: 40 }, (_, i) => i + 0.5)
    const s2 = [...s1.slice(1), s1[0]]

    expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
    expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
  })

  // Regression: the match table is a `Map`, which matches keys by
  // SameValueZero — under which `NaN` equals itself. Everything else here
  // compares with `===`, so the kernel used to report matches the affix
  // trimming and the scalar fallbacks did not.
  it('treats a nested NaN as matching nothing, as `===` does', () => {
    expect(lcsLength([NaN], [0, NaN, 0])).toBe(0)
    expect(levenshteinUniform([NaN], [0, NaN, 0])).toBe(3)
    expect(lcsLength([NaN, NaN], [NaN, NaN])).toBe(0)
    expect(levenshteinUniform([NaN, NaN], [NaN, NaN])).toBe(2)

    const token = fc.constantFrom(NaN, 1, 2, 300, 'x')
    fc.assert(
      fc.property(fc.array(token), fc.array(token), (s1, s2) => {
        expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
        expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
      }),
      { numRuns: 300 },
    )
  })

  it('survives a fractional miss budget, which cannot index the mbleven table', () => {
    const s1 = [1, 2, 3, 4]
    const s2 = [1, 2, 4]

    for (const budget of [0.5, 2.5, 4.5, Number.NaN, Infinity]) {
      expect(
        lcsLengthRange(s1, 0, s1.length, s2, 0, s2.length, budget),
      ).toBeLessThanOrEqual(lcsReference(s1, s2))
    }

    // A budget comfortably above the true distance still has to be exact.
    expect(lcsLengthRange(s1, 0, s1.length, s2, 0, s2.length, 4.5)).toBe(
      lcsReference(s1, s2),
    )
  })
})

describe('alignment matrices agree with strict sequence equality', () => {
  it('does not match NaN with itself', () => {
    expect(lcsSeqMatrix([NaN], 0, 1, [NaN], 0, 1).sim).toBe(0)
    expect(levenshteinMatrix([NaN], 0, 1, [NaN], 0, 1).dist).toBe(1)
    expect(commonAffix([NaN], [NaN])).toEqual({ prefixLen: 0, suffixLen: 0 })
  })

  // The ported suite covers an empty *first* input, which returns before the
  // matrix is set up. An empty second one leaves a matrix with no rows, which
  // the shortcut has to score without walking a vector that was never built.
  it('scores an empty second input against a long first one', () => {
    const s1 = Array.from({ length: 300 }, (_, i) => i % 7)

    expect(lcsSeqMatrix(s1, 0, s1.length, [], 0, 0).sim).toBe(0)
    expect(levenshteinMatrix(s1, 0, s1.length, [], 0, 0).dist).toBe(s1.length)
    expect(editopTuples(levenshteinEditops('ab', ''))).toEqual([
      ['delete', 0, 0],
      ['delete', 1, 0],
    ])
    expect(editopTuples(lcsSeqEditops('ab', ''))).toEqual([
      ['delete', 0, 0],
      ['delete', 1, 0],
    ])
  })

  // The matrices index a pattern's match masks by the element itself, over the
  // span between its least and greatest. Everything else — a fraction, which is
  // no index into a typed array; a negative number; an element that is not a
  // number at all; a span too wide to be worth a table — has to reach the same
  // answer through the map beside it. Both widths, because the one-word and the
  // multi-word builder partition their elements separately.
  it('scores elements the span table cannot index', () => {
    const marker = { position: 'not a number' }
    const cases: Array<[unknown[], unknown[]]> = [
      [
        [1.5, 2.5, 1.5, 4],
        [1.5, 4, 2.5],
      ],
      // A fraction *inside* the span of the integers around it: the table has
      // no element at index 1.5, so a mask written there would be dropped and
      // read back as `undefined` — the element would match nothing, itself
      // included.
      [
        [1, 2.5, 4, 2.5],
        [2.5, 4, 1],
      ],
      [
        [1, 2, 3.5, 4, 3.5, 2],
        [3.5, 1, 3.5, 4],
      ],
      [
        [-1, -2, -1, 3],
        [-1, 3, -2],
      ],
      [
        [marker, 1, marker],
        [1, marker, 1],
      ],
      [
        ['a', 'b', 'a'],
        ['b', 'a', 'b'],
      ],
      [
        [0, 1_000_000, 0, 7],
        [1_000_000, 7, 0],
      ],
      [
        [Number.NaN, 1, Number.NaN],
        [1, Number.NaN, 1],
      ],
    ]

    for (const [short1, short2] of cases) {
      // Four times the width, so the same elements run through the multi-word
      // builder as well, where each of them owns a slice of one flat array.
      const long1 = [...short1, ...short1, ...short1, ...short1].flatMap((e) => [e, e, e])
      const long2 = [...short2, ...short2, ...short2, ...short2].flatMap((e) => [e, e, e])

      for (const [a, b] of [
        [short1, short2],
        [long1, long2],
      ]) {
        expect(lcsSeqMatrix(a, 0, a.length, b, 0, b.length).sim).toBe(lcsReference(a, b))
        expect(levenshteinMatrix(a, 0, a.length, b, 0, b.length).dist).toBe(
          levenshteinReference(a, b),
        )
      }
    }
  })

  // The multi-word builder sizes its table from the span, or from a cap when
  // the span reaches past one, and grows it when a range holds more distinct
  // elements than that. Growth copies what is already filed, so a lost copy
  // shows as elements that stop matching partway through the range.
  it('files every element of a range whose span reaches past the cap', () => {
    // A span of 4000 the range fills 400 of, so the table stays at its cap.
    const sparse = Array.from({ length: 400 }, (_, i) => (i % 8) * 500)
    // A span of 400 the range fills every entry of, so the table grows to it.
    const dense = Array.from({ length: 400 }, (_, i) => i)
    // The same, plus elements no table can index: those draw blocks from the
    // table too, so it has to grow again once the span is used up.
    const beyond = [...dense, ...Array.from({ length: 300 }, (_, i) => `s${i}`)]

    for (const s1 of [sparse, dense, beyond]) {
      for (const s2 of [s1, s1.filter((_, i) => i % 3 !== 0)]) {
        expect(lcsSeqMatrix(s1, 0, s1.length, s2, 0, s2.length).sim).toBe(
          lcsReference(s1, s2),
        )
        expect(levenshteinMatrix(s1, 0, s1.length, s2, 0, s2.length).dist).toBe(
          levenshteinReference(s1, s2),
        )
      }
    }
  })

  it('files every element of a range that outgrows its first table', () => {
    // A span far wider than the range is no table at all, so all 100 of these
    // are filed in the map — well past the blocks the table starts with.
    const spread = Array.from({ length: 100 }, (_, i) => i * 1000)
    // An accepted span for the integers, plus enough elements no table can
    // index that the blocks behind them outgrow it too.
    const mixed = [
      ...Array.from({ length: 40 }, (_, i) => i),
      ...Array.from({ length: 80 }, (_, i) => `s${i}`),
    ]

    for (const s1 of [spread, mixed]) {
      // The range against itself, because that is what a dropped block shows
      // in: every element filed before the table grew has to still match, and
      // an element that matches nothing costs the alignment two operations.
      for (const s2 of [s1, s1.filter((_, i) => i % 3 !== 0)]) {
        expect(lcsSeqMatrix(s1, 0, s1.length, s2, 0, s2.length).sim).toBe(
          lcsReference(s1, s2),
        )
        expect(levenshteinMatrix(s1, 0, s1.length, s2, 0, s2.length).dist).toBe(
          levenshteinReference(s1, s2),
        )
      }
    }
  })

  // An element the range does not hold has no masks of its own and reads the
  // block reserved for absent ones, which stays zeroed. Both kinds have to
  // reach it: one inside the span the table covers, one outside it entirely.
  it('matches nothing against elements the pattern does not hold', () => {
    const s1 = Array.from({ length: 130 }, (_, i) => 2 * (i % 40))
    const inSpan = Array.from({ length: 90 }, (_, i) => 2 * (i % 40) + 1)
    const outside = Array.from({ length: 90 }, (_, i) => 500 + (i % 40))

    for (const s2 of [inSpan, outside, [...inSpan, ...outside]]) {
      expect(lcsSeqMatrix(s1, 0, s1.length, s2, 0, s2.length).sim).toBe(0)
      expect(levenshteinMatrix(s1, 0, s1.length, s2, 0, s2.length).dist).toBe(
        levenshteinReference(s1, s2),
      )
    }
  })

  // The `word boundaries` block below covers these lengths for the scoring
  // kernels. The matrices have their own builder and their own row store, and
  // only ever met a word boundary here at 257 elements.
  it.each([33, 63, 64, 65, 127, 128, 129])(
    'aligns a range of exactly %i elements',
    (length) => {
      const s1 = Array.from({ length }, (_, i) => 97 + ((i * 7) % 5))
      const s2 = Array.from({ length: length + 11 }, (_, i) => 97 + ((i * 3) % 5))

      expect(lcsSeqMatrix(s1, 0, s1.length, s2, 0, s2.length).sim).toBe(
        lcsReference(s1, s2),
      )
      expect(levenshteinMatrix(s1, 0, s1.length, s2, 0, s2.length).dist).toBe(
        levenshteinReference(s1, s2),
      )
    },
  )

  // The Hirschberg dispatch asks how big a matrix would be before building it,
  // and the estimate it used was written for row objects that no longer exist —
  // it charged 24 bytes of overhead per row and counted the band in bits, so on
  // a narrow band it read two to three times high. Sharing the sizing with the
  // allocation is what keeps the two from drifting apart again, so the test is
  // that they agree exactly rather than that the number is any given value.
  it.each([
    [64, 40, -1],
    [64, 40, 4],
    [100, 70, 15],
    [100, 70, 16],
    [100, 70, 49],
    [100, 70, 50],
    [257, 311, 8],
    [257, 311, 128],
    [32, 20, 4],
    [33, 20, 4],
  ])('sizes a %i x %i matrix at distance %i as it allocates it', (len1, len2, budget) => {
    const s1 = Array.from({ length: len1 }, (_, i) => 97 + ((i * 7) % 5))
    const s2 = Array.from({ length: len2 }, (_, i) => 97 + ((i * 3) % 5))
    const { vp, vn, offsets } = levenshteinMatrix(s1, 0, len1, s2, 0, len2, budget)

    expect(levenshteinMatrixBytes(len1, len2, budget)).toBe(
      vp.byteLength + vn.byteLength + (offsets === null ? 0 : offsets.byteLength),
    )
  })

  it('counts long final vectors without losing high bits', () => {
    const s1 = Array.from({ length: 257 }, (_, i) => i % 11)
    const s2 = Array.from({ length: 311 }, (_, i) => (i * 7) % 13)

    expect(lcsSeqMatrix(s1, 0, s1.length, s2, 0, s2.length).sim).toBe(
      lcsReference(s1, s2),
    )
    expect(levenshteinMatrix(s1, 0, s1.length, s2, 0, s2.length).dist).toBe(
      levenshteinReference(s1, s2),
    )
  })

  // The matrices read a range of each input rather than a copy of it, so a
  // window into a longer sequence has to score as the window alone would.
  it('scores a range exactly as the isolated subsequence does', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 24 }),
        fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 24 }),
        fc.array(fc.integer({ min: 0, max: 4 }), { maxLength: 8 }),
        fc.array(fc.integer({ min: 0, max: 4 }), { maxLength: 8 }),
        (a, b, before, after) => {
          const paddedA = [...before, ...a, ...after]
          const paddedB = [...before, ...b, ...after]

          expect(
            levenshteinMatrix(
              paddedA,
              before.length,
              a.length,
              paddedB,
              before.length,
              b.length,
            ).dist,
          ).toBe(levenshteinMatrix(a, 0, a.length, b, 0, b.length).dist)
          expect(
            lcsSeqMatrix(
              paddedA,
              before.length,
              a.length,
              paddedB,
              before.length,
              b.length,
            ).sim,
          ).toBe(lcsSeqMatrix(a, 0, a.length, b, 0, b.length).sim)
        },
      ),
      { numRuns: 300 },
    )
  })

  it('retains the same Levenshtein bits inside a narrow recovery band', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 48 }),
        fc.array(fc.integer({ min: 0, max: 4 }), { minLength: 1, maxLength: 48 }),
        fc.integer({ min: 0, max: 12 }),
        (a, b, maximum) => {
          const full = levenshteinMatrix(a, 0, a.length, b, 0, b.length)
          const banded = levenshteinMatrix(a, 0, a.length, b, 0, b.length, maximum)
          expect(banded.dist).toBe(full.dist)
          const offsets = banded.offsets
          for (let row = 0; row < b.length; row++) {
            const firstColumn = Math.max(1, row + 1 - maximum)
            const lastColumn = Math.min(a.length, row + 1 + maximum)
            for (let column = firstColumn; column <= lastColumn; column++) {
              const expectedVp = rowBitSet(full.vp, full.stride, row, column - 1)
              const expectedVn = rowBitSet(full.vn, full.stride, row, column - 1)
              if (offsets === null) {
                expect(rowBitSet(banded.vp, banded.stride, row, column - 1)).toBe(
                  expectedVp,
                )
                expect(rowBitSet(banded.vn, banded.stride, row, column - 1)).toBe(
                  expectedVn,
                )
                continue
              }
              expect(
                shiftedRowBitSet(banded.vp, banded.stride, row, offsets[row], column - 1),
              ).toBe(expectedVp)
              expect(
                shiftedRowBitSet(banded.vn, banded.stride, row, offsets[row], column - 1),
              ).toBe(expectedVn)
            }
          }
        },
      ),
      { numRuns: 300 },
    )
  })

  // A row keeps only its band, stored from the word boundary at or below the
  // band's first position — so the same band needs one more word when it starts
  // 31 bits into that word than when it starts on it. The property above never
  // sees this: its inputs are at most two words wide. These maxima put a band
  // either side of each word boundary, and `b` is long enough that the last
  // rows have their window clamped against the end of the vector.
  it('keeps a multi-word band wherever inside a word it starts', () => {
    const a = Array.from({ length: 200 }, (_, i) => (i * 7) % 5)
    const b = Array.from({ length: 260 }, (_, i) => (i * 11) % 5)
    const full = levenshteinMatrix(a, 0, a.length, b, 0, b.length)

    for (const maximum of [0, 1, 15, 16, 17, 32, 33]) {
      const banded = levenshteinMatrix(a, 0, a.length, b, 0, b.length, maximum)
      const offsets = banded.offsets
      if (offsets === null) throw new Error(`band ${maximum} was stored whole`)
      expect(banded.dist).toBe(full.dist)

      for (let row = 0; row < b.length; row++) {
        const firstColumn = Math.max(1, row + 1 - maximum)
        const lastColumn = Math.min(a.length, row + 1 + maximum)
        for (let column = firstColumn; column <= lastColumn; column++) {
          expect(
            shiftedRowBitSet(banded.vp, banded.stride, row, offsets[row], column - 1),
          ).toBe(rowBitSet(full.vp, full.stride, row, column - 1))
          expect(
            shiftedRowBitSet(banded.vn, banded.stride, row, offsets[row], column - 1),
          ).toBe(rowBitSet(full.vn, full.stride, row, column - 1))
        }
      }
    }
  })
})

describe('word boundaries', () => {
  // 31/32/33 straddle the single-word limit; 63/64/65 the second word; 96 is a
  // whole number of words with nothing left over.
  const lengths = [1, 31, 32, 33, 63, 64, 65, 96, 97, 200]

  it.each(lengths)('handles a pattern of exactly %i elements', (length) => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 99 }), {
          minLength: length,
          maxLength: length,
        }),
        fc.array(fc.integer({ min: 97, max: 99 }), { maxLength: 120 }),
        (s1, s2) => {
          expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
          expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
        },
      ),
      { numRuns: 60 },
    )
  })

  // Regression: the multi-word match table hands each distinct element an
  // offset into one shared pool, and the pool has to grow once a pattern holds
  // more distinct elements than it was sized for. Growing it used to allocate a
  // fresh buffer without copying, which dropped every mask built so far and
  // left the already-issued offsets pointing at zeros. Reaching it needs both a
  // multi-word pattern and an alphabet wider than the pool's initial capacity,
  // which no small-alphabet test produces.
  it.each([64, 65, 200, 400, 600])(
    'keeps earlier masks when the pool grows for %i distinct elements',
    (distinct) => {
      // Above the direct-lookup range, so every element needs a pool slot.
      const s1 = Array.from({ length: distinct }, (_, i) => 0x3000 + i)
      const s2 = [...s1.slice(1), s1[0]]

      expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
      expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
    },
  )

  it('handles two long strings differing only in the middle', () => {
    // Common-affix trimming shortens the pattern; this is the shape where the
    // trimmed middle and the untrimmed input land in different kernels.
    const head = 'a'.repeat(100)
    const tail = 'z'.repeat(100)
    const s1 = codePoints(`${head}abcabc${tail}`)
    const s2 = codePoints(`${head}cbacba${tail}`)

    expect(lcsLength(s1, s2)).toBe(lcsReference(s1, s2))
    expect(levenshteinUniform(s1, s2)).toBe(levenshteinReference(s1, s2))
  })
})

describe('the OSA kernel agrees with its dynamic program', () => {
  it('handles an empty pattern in the exported one-word kernel', () => {
    expect(osaOneWord([], [1, 2])).toBe(2)
    expect(osaOneWord([], [])).toBe(0)
  })

  it('on small alphabets, where transpositions actually occur', () => {
    fc.assert(
      fc.property(smallAlphabet, smallAlphabet, (a, b) => {
        const s1 = codePoints(a)
        const s2 = codePoints(b)
        expect(osaDistance(s1, s2)).toBe(osaReference(s1, s2))
      }),
      { numRuns: 500 },
    )
  })

  it('agrees across the fallback boundary', () => {
    // Above one word `osaDistance` runs the dynamic program instead of the
    // kernel. Both sides of that switch have to produce the same number.
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 99 }), { minLength: 25, maxLength: 40 }),
        fc.array(fc.integer({ min: 97, max: 99 }), { minLength: 25, maxLength: 40 }),
        (s1, s2) => {
          expect(osaDistance(s1, s2)).toBe(osaReference(s1, s2))
        },
      ),
      { numRuns: 300 },
    )
  })

  it('keeps OSA distinct from Damerau-Levenshtein', () => {
    // The transposition term must not turn OSA into unrestricted Damerau: OSA
    // forbids editing a substring twice, so this pair is 3 and not 2.
    expect(osaDistance('CA', 'ABC')).toBe(3)
  })
})

describe('bounded kernels preserve unbounded results', () => {
  it('reads one-word wide masks in the small Levenshtein band', () => {
    expect(levenshteinDistance('aЖaa', 'ЖaaЖ', { scoreCutoff: 4, scoreHint: 4 })).toBe(2)

    const cases: ReadonlyArray<readonly [ArrayLike<unknown>, ArrayLike<unknown>]> = [
      ['aЖaa', 'ЖaaЖ'],
      ['a🙂aa', '🙂aa🙂'],
      [
        [0, 1.5, 0, 0],
        [1.5, 0, 0, 1.5],
      ],
      [
        ['a', 'wide', 'a', 'a'],
        ['wide', 'a', 'a', 'wide'],
      ],
    ]

    for (const [a, b] of cases) {
      const exact = levenshteinDistance(a, b)
      expect(levenshteinDistance(a, b, { scoreCutoff: 4, scoreHint: 4 })).toBe(exact)
    }
  })

  it('uses scoreHint only for algorithm selection', () => {
    fc.assert(
      fc.property(
        smallAlphabet,
        smallAlphabet,
        fc.integer({ min: 0, max: 20 }),
        (a, b, hint) => {
          const distance = levenshteinDistance(a, b)
          const similarity = levenshteinSimilarity(a, b)
          const normalizedDistance = levenshteinNormalizedDistance(a, b)
          const normalizedSimilarity = levenshteinNormalizedSimilarity(a, b)

          expect(levenshteinDistance(a, b, { scoreHint: hint })).toBe(distance)
          expect(levenshteinDistance(a, b, { scoreHint: distance })).toBe(distance)
          expect(levenshteinSimilarity(a, b, { scoreHint: hint })).toBe(similarity)
          expect(levenshteinNormalizedDistance(a, b, { scoreHint: hint / 20 })).toBe(
            normalizedDistance,
          )
          expect(levenshteinNormalizedSimilarity(a, b, { scoreHint: hint / 20 })).toBe(
            normalizedSimilarity,
          )
        },
      ),
      { numRuns: 500 },
    )
  })

  it('returns the exact distance inside the cutoff and the documented sentinel outside', () => {
    fc.assert(
      fc.property(
        smallAlphabet,
        smallAlphabet,
        fc.integer({ min: 0, max: 12 }),
        (a, b, cutoff) => {
          const exact = levenshteinDistance(a, b)
          expect(levenshteinDistance(a, b, { scoreCutoff: cutoff })).toBe(
            exact <= cutoff ? exact : cutoff + 1,
          )

          const damerau = damerauLevenshteinDistance(a, b)
          expect(damerauLevenshteinDistance(a, b, { scoreCutoff: cutoff })).toBe(
            damerau <= cutoff ? damerau : cutoff + 1,
          )
        },
      ),
      { numRuns: 500 },
    )
  })

  it('keeps Jaro cutoff boundaries identical to thresholding the exact score', () => {
    fc.assert(
      fc.property(
        smallAlphabet,
        smallAlphabet,
        fc.double({ min: 0, max: 1, noNaN: true }),
        (a, b, cutoff) => {
          const exact = jaroSimilarity(a, b)
          expect(jaroSimilarity(a, b, { scoreCutoff: cutoff })).toBe(
            exact >= cutoff ? exact : 0,
          )
        },
      ),
      { numRuns: 500 },
    )
  })
})

describe('weighted Levenshtein fast paths', () => {
  it('match the generic dynamic program in either orientation', () => {
    const scaledIndel: LevenshteinWeights = [2, 2, 5]
    const expensiveDelete: LevenshteinWeights = [3, 7, 5]
    const expensiveInsert: LevenshteinWeights = [7, 3, 5]
    const scaledUniform: LevenshteinWeights = [2, 2, 2]
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 3 }), { maxLength: 16 }),
        fc.array(fc.integer({ min: 0, max: 3 }), { maxLength: 16 }),
        fc.constantFrom(scaledIndel, expensiveDelete, expensiveInsert, scaledUniform),
        (a, b, weights) => {
          const expected = weightedLevenshteinReference(a, b, ...weights)
          expect(levenshteinDistance(a, b, { weights })).toBe(expected)
          for (const cutoff of [Math.max(0, expected - 1), expected, expected + 1]) {
            expect(levenshteinDistance(a, b, { weights, scoreCutoff: cutoff })).toBe(
              expected <= cutoff ? expected : cutoff + 1,
            )
          }
        },
      ),
      { numRuns: 500 },
    )
  })

  // The generic DP has two kernels: whole non-negative weights run over an
  // `Int32Array` with a finite out-of-band sentinel, anything else over a
  // `Float64Array` with `Infinity`. The pair above only reaches the integer
  // one, so these cover the float kernel and the choice between them.
  it('agree between the integer and floating-point kernels', () => {
    // Non-negative weights only: the affix trimming every path shares assumes
    // editing a matching pair can never pay, as upstream does.
    const fractional: LevenshteinWeights = [1.5, 2.25, 3.5]
    const mixed: LevenshteinWeights = [2, 3, 4.5]
    const huge: LevenshteinWeights = [2 ** 29, 2 ** 29, 2 ** 29 + 1]
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 3 }), { maxLength: 16 }),
        fc.array(fc.integer({ min: 0, max: 3 }), { maxLength: 16 }),
        fc.constantFrom(fractional, mixed, huge),
        (a, b, weights) => {
          expect(levenshteinDistance(a, b, { weights })).toBe(
            weightedLevenshteinReference(a, b, ...weights),
          )
        },
      ),
      { numRuns: 400 },
    )
  })

  // A cost near the integer kernel's ceiling must either stay exact or hand
  // over to the float one — never wrap, and never collide with the sentinel.
  it('stay exact for weights close to the integer kernel ceiling', () => {
    const a = [0, 1, 2, 3, 0, 1]
    const b = [3, 2, 1, 0, 3, 2, 1]

    for (const weight of [2 ** 20, 2 ** 26, 2 ** 28, 2 ** 29, 2 ** 30, 2 ** 31]) {
      const weights: LevenshteinWeights = [weight, weight, weight + 1]
      expect(levenshteinDistance(a, b, { weights })).toBe(
        weightedLevenshteinReference(a, b, ...weights),
      )
    }
  })
})

describe('the score-cutoff prune never changes a result', () => {
  // `ratio` rejects candidates whose lengths alone put them out of reach of the
  // cutoff, without running the kernel. If that ceiling were ever wrong — or
  // merely compared with a lookalike predicate — a pair would be dropped that
  // should have scored, and only at cutoffs that land exactly on a score.
  it('agrees with scoring first and thresholding afterwards', () => {
    fc.assert(
      fc.property(
        smallAlphabet,
        smallAlphabet,
        fc.double({ min: 0, max: 100, noNaN: true }),
        (a, b, cutoff) => {
          const unpruned = ratio(a, b)
          const expected = unpruned >= cutoff ? unpruned : 0
          expect(ratio(a, b, { scoreCutoff: cutoff })).toBe(expected)
        },
      ),
      { numRuns: 1000 },
    )
  })

  it('returns either the exact score or zero, never something in between', () => {
    fc.assert(
      fc.property(
        smallAlphabet,
        smallAlphabet,
        fc.double({ min: 0, max: 100, noNaN: true }),
        (a, b, cutoff) => {
          const pruned = ratio(a, b, { scoreCutoff: cutoff })
          expect(pruned === 0 || pruned === ratio(a, b)).toBe(true)
        },
      ),
      { numRuns: 1000 },
    )
  })

  it('never rejects a pair that clears the cutoff, on very different lengths', () => {
    // Where the prune fires hardest. A margin of 0.5 keeps this clear of the
    // scaling edge documented below, which is about float representation
    // rather than about the prune.
    fc.assert(
      fc.property(
        fc.string({ minLength: 0, maxLength: 4 }),
        fc.string({ minLength: 20, maxLength: 60 }),
        (a, b) => {
          const score = ratio(a, b)
          if (score > 0.5) expect(ratio(a, b, { scoreCutoff: score - 0.5 })).toBe(score)
          expect(ratio(a, b, { scoreCutoff: score + 0.5 })).toBe(0)
        },
      ),
      { numRuns: 500 },
    )
  })
})

describe('scoreCutoff scaling matches upstream, including where it is lossy', () => {
  it('drops a score when the cutoff is its own value', () => {
    // Not a defect of ours: `ratio` divides the 0-100 cutoff by 100 and
    // compares against an unscaled similarity, exactly as fuzz_py.ratio does.
    // The round-trip through x100 is lossy, so a cutoff set to a score can land
    // one ULP above it and reject its own value.
    //
    // Verified against upstream's pure-Python backend, which returns 0 here
    // too. Pinned because it looks like a bug and must not be "fixed" into a
    // difference from RapidFuzz.
    const score = ratio('baaaa', 'cbbaaaa')

    expect(score).toBe(83.33333333333334)
    expect(ratio('baaaa', 'cbbaaaa', { scoreCutoff: score })).toBe(0)
  })
})

describe('scratch buffers are not shared across calls', () => {
  it('gives the same answer whichever order the pairs are scored in', () => {
    // The kernels reuse module-level buffers. If one call could observe
    // another's state, interleaving would change the answers.
    const pairs: Array<[number[], number[]]> = [
      [codePoints('abcabcabc'), codePoints('cbacbacba')],
      [codePoints('x'.repeat(70)), codePoints('y'.repeat(70))],
      [codePoints('hello world'), codePoints('hallo welt')],
      [codePoints('😀'.repeat(40)), codePoints('😁'.repeat(40))],
    ]

    const alone = pairs.map(([a, b]) => lcsLength(a, b))
    const interleaved = pairs.map(([a, b], i) => {
      const other = pairs[(i + 1) % pairs.length]
      lcsLength(other[0], other[1])
      return lcsLength(a, b)
    })

    expect(interleaved).toEqual(alone)
  })

  // The multi-word kernels keep the last pattern's masks in the shared table
  // and skip rebuilding them when the same pattern comes back, which is what
  // makes scoring one query against a list cheap. Reuse is only sound while
  // nothing else has written the table since, and only for patterns that
  // cannot have changed meanwhile.
  it('rebuilds a pattern the table no longer holds', () => {
    const query = 'abcdefghij'.repeat(12)
    const other = 'zyxwvutsrq'.repeat(12)
    const choices = ['abcdefghij'.repeat(12), 'abcdefghij'.repeat(11), other]

    for (const choice of choices) {
      const expected = levenshteinReference(codePoints(query), codePoints(choice))

      // Interpose a different pattern between two runs of the same one.
      expect(levenshteinDistance(query, choice)).toBe(expected)
      levenshteinDistance(other, choice)
      expect(levenshteinDistance(query, choice)).toBe(expected)
    }
  })

  it('does not reuse masks for a sequence that could have changed', () => {
    const pattern = Array.from('abcdefghij'.repeat(12), (c) => c.codePointAt(0))
    const choice = codePoints('abcdefghij'.repeat(11))

    const original = [...pattern]

    expect(levenshteinUniform(pattern, choice)).toBe(
      levenshteinReference(original, choice),
    )

    pattern[0] = 0x21
    expect(levenshteinUniform(pattern, choice)).toBe(
      levenshteinReference(pattern, choice),
    )
  })
})

// A cutoff sends `levenshteinUniform` down a different kernel entirely — the
// Ukkonen-banded blocked Myers, which walks only the words the budget can still
// reach and tightens that budget after every row. The tests above all call it
// without a cutoff, so none of them reach the banded kernel at all.
describe('the banded Levenshtein kernel agrees with the dynamic program', () => {
  /** What a bounded run promises: the distance, or anything above the budget. */
  function expectBounded(
    s1: ArrayLike<unknown>,
    s2: ArrayLike<unknown>,
    budget: number,
  ): void {
    const expected = levenshteinReference(s1, s2)
    const actual = levenshteinUniform(s1, s2, budget)

    if (expected <= budget) expect(actual).toBe(expected)
    else expect(actual).toBeGreaterThan(budget)
  }

  // Budgets either side of every dispatch boundary: 3/4 hands mbleven over to
  // the small band, 31/32 hands the small band over to the blocked band.
  const budgets = [0, 1, 3, 4, 8, 31, 32, 33, 64, 65, 96, 200, 1000]

  it.each(budgets)('is exact at a budget of %i', (budget) => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 100 }), { maxLength: 150 }),
        fc.array(fc.integer({ min: 97, max: 100 }), { maxLength: 150 }),
        (s1, s2) => {
          expectBounded(s1, s2, budget)
        },
      ),
      { numRuns: 120 },
    )
  })

  // A budget at or above the longest input can reject nothing, so the dispatch
  // drops the band entirely and runs the unbounded kernel. That equivalence is
  // the whole point of the clamp, and it is invisible to a smaller budget.
  it('matches the unbounded kernel once the budget stops rejecting', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 101 }), { maxLength: 120 }),
        fc.array(fc.integer({ min: 97, max: 101 }), { maxLength: 120 }),
        (s1, s2) => {
          const longest = Math.max(s1.length, s2.length)
          const expected = levenshteinUniform(s1, s2)

          expect(levenshteinUniform(s1, s2, longest)).toBe(expected)
          expect(levenshteinUniform(s1, s2, longest + 1)).toBe(expected)
          expect(levenshteinUniform(s1, s2, Number.MAX_SAFE_INTEGER)).toBe(expected)
        },
      ),
      { numRuns: 200 },
    )
  })

  // The band is measured in whole words, so its arithmetic only shows itself
  // on patterns long enough to hold several — the shapes below are the ones a
  // 150-element random case reaches too rarely to be trusted.
  it.each([64, 96, 128, 200, 400])(
    'stays exact on %i-element inputs across a range of budgets',
    (length) => {
      fc.assert(
        fc.property(
          fc.array(fc.integer({ min: 97, max: 99 }), {
            minLength: length,
            maxLength: length,
          }),
          fc.array(fc.integer({ min: 97, max: 99 }), {
            minLength: length - 20,
            maxLength: length + 20,
          }),
          fc.integer({ min: 0, max: 300 }),
          (s1, s2, budget) => {
            expectBounded(s1, s2, budget)
          },
        ),
        { numRuns: 150 },
      )
    },
  )

  // Regression: the diagonal band is `2 * budget + 1` wide and has to fit in
  // one word, but the dispatch only checked the budget — so budgets from 16 to
  // 31 ran the small-band kernel with a band twice what it could hold, and the
  // part that did not fit was quietly dropped. This pair is 20 edits apart and
  // came back as 21 at every budget in that range that could afford it.
  it.each([16, 20, 25, 30, 31])('keeps the whole band at a budget of %i', (budget) => {
    const s1 = codePoints(
      'aaabababaaaaaabababcaaacbaaaaaaacbbaaaaaaaaaaaabbaaaaaaacbababba',
    )
    const s2 = codePoints('a'.repeat(36) + 'b' + 'a'.repeat(13) + 'b' + 'a'.repeat(9))

    expect(s1.length).toBe(64)
    expect(s2.length).toBe(60)
    expect(levenshteinReference(s1, s2)).toBe(20)
    expectBounded(s1, s2, budget)
  })

  // Regression: the same band, one dispatch boundary further in. The width was
  // then tested as `min(len1, 2 * budget + 1) <= 32`, which is upstream's own
  // test — but upstream only reaches it once the whole matrix has failed to fit
  // a word, so its `min` never picks `len1`. Ported without that ordering it
  // did, and an input of a word or less admitted a band of any width at all.
  // This pair is 23 edits apart and came back as 24 at a budget of 30.
  it.each([16, 20, 25, 30, 31])(
    'keeps the whole band on a word-long input at a budget of %i',
    (budget) => {
      const s1 = codePoints('dbbbbddcadcdadbbcbdcbddadbbaacad')
      const s2 = codePoints('bddbacdbbcabbccbbacaacccabbddabdb')

      expect(s1.length).toBe(32)
      expect(s2.length).toBe(33)
      expect(levenshteinReference(s1, s2)).toBe(23)
      expectBounded(s1, s2, budget)
    },
  )

  // What let that through: a budget over 15 only reaches the one-word matrix
  // when both inputs sit between the budget and a word in length, and the
  // random cases above draw lengths up to 150, so they land there about once in
  // a hundred — too rarely to have failed reliably.
  it.each([16, 20, 25, 31])(
    'is exact on word-long inputs at a budget of %i',
    (budget) => {
      const side = fc.array(fc.integer({ min: 97, max: 100 }), {
        minLength: budget,
        maxLength: 33,
      })

      fc.assert(
        fc.property(side, side, (s1, s2) => {
          expectBounded(s1, s2, budget)
        }),
        { numRuns: 300 },
      )
    },
  )

  // A hint only picks the width the search starts at, so no hint may ever
  // change an answer — including hints below, at and above the true distance.
  it('gives the same answer whatever the hint', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 100 }), { maxLength: 140 }),
        fc.array(fc.integer({ min: 97, max: 100 }), { maxLength: 140 }),
        fc.integer({ min: 0, max: 200 }),
        (s1, s2, budget) => {
          const expected = levenshteinUniform(s1, s2, budget)

          for (const hint of [0, 1, 5, 31, 32, 64, 150, Number.MAX_SAFE_INTEGER]) {
            expect(levenshteinUniform(s1, s2, budget, hint)).toBe(expected)
          }
        },
      ),
      { numRuns: 120 },
    )
  })

  // Length-skewed pairs put the answer's diagonal far from the band's start,
  // which is what the band-widening step exists for.
  it('handles inputs of very different lengths', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 99 }), { maxLength: 20 }),
        fc.array(fc.integer({ min: 97, max: 99 }), { minLength: 100, maxLength: 300 }),
        fc.integer({ min: 0, max: 320 }),
        (short, long, budget) => {
          expectBounded(short, long, budget)
          expectBounded(long, short, budget)
        },
      ),
      { numRuns: 150 },
    )
  })

  it('is exact on elements that miss the direct lookup table', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom('a', 'β', '€', '😀', NaN, { id: 1 }), {
          maxLength: 120,
        }),
        fc.array(fc.constantFrom('a', 'β', '€', '😀', NaN, { id: 1 }), {
          maxLength: 120,
        }),
        fc.integer({ min: 0, max: 130 }),
        (s1, s2, budget) => {
          expectBounded(s1, s2, budget)
        },
      ),
      { numRuns: 200 },
    )
  })
})

/** A reproducible generator, so a failure names a case that can be re-run. */
function makeRng(seed: number): () => number {
  let state = seed >>> 0

  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0x1_0000_0000
  }
}

/** `length` elements drawn from an alphabet of `alphabet` code points. */
function randomElements(rng: () => number, length: number, alphabet: number): number[] {
  return Array.from({ length }, () => 97 + Math.floor(rng() * alphabet))
}

/**
 * A copy of `pattern` at `length` elements, with about one element in ten
 * changed.
 *
 * Two independently drawn sequences are usually far enough apart that the whole
 * interesting range of targets is already unreachable, and a bounded kernel
 * that refuses everything passes that. A near-copy puts the true length in the
 * middle of the sweep, where a band that drops a word actually shows.
 */
function nearCopy(
  rng: () => number,
  pattern: readonly number[],
  length: number,
): number[] {
  return Array.from({ length }, (_unused, i) => {
    const source = i < pattern.length ? pattern[i] : 97 + Math.floor(rng() * 4)
    return rng() < 0.1 ? source + 1 : source
  })
}

// The prepared LCS kernels read masks built once and held, so they cannot trim
// a common affix and cannot be reached through any scorer that would. Nothing
// called them directly until this block: they were covered only as far as
// `ratio`, `extract` and the prepared scorers happened to reach them, which is
// not far enough for a kernel that decides for itself which words to look at.
//
// `lcsLengthPreparedBounded` in particular promises something stronger than the
// range kernel does — not "no greater than the true length" but "negative, or
// exactly right" — and its band is free to leave whole words unvisited. The
// dangerous failure is therefore silent: a reachable subsequence reported as
// out of reach. That is what `expectBounded` below asserts in both directions.
describe('the prepared LCS kernels agree with the dynamic program', () => {
  /** The exact kernel owes the dynamic program's answer, always. */
  function expectExact(pattern: ArrayLike<unknown>, text: ArrayLike<unknown>): void {
    const prepared = preparePattern(pattern, 0, pattern.length)
    expect(lcsLengthPrepared(prepared, text, 0, text.length)).toBe(
      lcsReference(pattern, text),
    )
  }

  /**
   * What the bounded kernel promises: a negative result only when `required` is
   * genuinely out of reach, and the exact length whenever it is not.
   */
  function expectBounded(
    pattern: ArrayLike<unknown>,
    text: ArrayLike<unknown>,
    required: number,
  ): void {
    const expected = lcsReference(pattern, text)
    const prepared = preparePattern(pattern, 0, pattern.length)
    const actual = lcsLengthPreparedBounded(prepared, text, 0, text.length, required)

    // The band may only discard a pair the caller has already rejected.
    if (expected >= required) expect(actual).toBe(expected)
    else expect(actual < 0 || actual === expected).toBe(true)
  }

  /** Every `required` the caller can ask for, including one past the ceiling. */
  function sweepRequired(pattern: ArrayLike<unknown>, text: ArrayLike<unknown>): void {
    const ceiling = Math.min(pattern.length, text.length)
    for (let required = 0; required <= ceiling + 1; required++) {
      expectBounded(pattern, text, required)
    }
  }

  // Either side of every word boundary the masks change width at, and past the
  // three widths the multi-word kernel writes out before it loops.
  const lengths = [0, 1, 8, 31, 32, 33, 63, 64, 65, 96, 97, 128, 200]

  it.each(lengths)('is exact on a %i-element pattern, over every target', (length) => {
    const rng = makeRng(0x1c5 + length)

    for (const textLength of lengths) {
      const pattern = randomElements(rng, length, 4)
      const text = randomElements(rng, textLength, 4)

      expectExact(pattern, text)
      sweepRequired(pattern, text)
      // A near-copy puts the true length near the middle of the sweep, which is
      // where a band that drops a word shows itself. A pattern and text drawn
      // independently are usually far enough apart that every interesting
      // `required` is already unreachable.
      expectExact(pattern, nearCopy(rng, pattern, textLength))
      sweepRequired(pattern, nearCopy(rng, pattern, textLength))
    }
  })

  // A wide alphabet keeps the answer small, which pushes the band narrow; a
  // two-element one keeps it near the ceiling, which pushes the band wide.
  it.each([2, 4, 26, 200])('is exact over an alphabet of %i elements', (alphabet) => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 97, max: 96 + alphabet }), { maxLength: 180 }),
        fc.array(fc.integer({ min: 97, max: 96 + alphabet }), { maxLength: 180 }),
        fc.integer({ min: 0, max: 180 }),
        (pattern, text, required) => {
          expectExact(pattern, text)
          expectBounded(pattern, text, required)
        },
      ),
      { numRuns: 200 },
    )
  })

  // The masks address Latin-1 directly, high elements through a window and
  // everything else through a Map, and the band has to agree with all three.
  it('is exact on elements that miss the direct mask region', () => {
    const token = fc.constantFrom('a', 'β', '€', '😀', NaN, { id: 1 }, 0.5, 1200)

    fc.assert(
      fc.property(
        fc.array(token, { maxLength: 140 }),
        fc.array(token, { maxLength: 140 }),
        fc.integer({ min: 0, max: 140 }),
        (pattern, text, required) => {
          expectExact(pattern, text)
          expectBounded(pattern, text, required)
        },
      ),
      { numRuns: 250 },
    )
  })

  // A single script spans few enough code points to be windowed; two distant
  // ones do not, and every element takes a stray slot instead. The band indexes
  // the masks the same way in both, so both have to be swept.
  it.each([
    ['Cyrillic', 0x0410],
    ['Greek', 0x0391],
    ['CJK', 0x4e00],
  ])('is exact on a %s pattern', (_name, base) => {
    const rng = makeRng(base)

    for (const length of [33, 65, 130]) {
      const pattern = Array.from({ length }, () => base + Math.floor(rng() * 24))
      const text = Array.from(
        { length: length + 10 },
        () => base + Math.floor(rng() * 24),
      )

      expectExact(pattern, text)
      sweepRequired(pattern, text)
    }
  })

  // A target above either input is unreachable before an element is read, and
  // the band widths would go negative if it were not answered first.
  it('refuses a target no common subsequence could reach', () => {
    const pattern = Array.from({ length: 100 }, (_unused, i) => 97 + (i % 3))
    const text = pattern.slice(0, 40)

    expect(
      lcsLengthPreparedBounded(preparePattern(pattern, 0, 100), text, 0, 40, 41),
    ).toBeLessThan(0)
    expect(
      lcsLengthPreparedBounded(preparePattern(pattern, 0, 100), text, 0, 40, 101),
    ).toBeLessThan(0)
    // Exactly at the ceiling is reachable in principle, so it must be answered
    // rather than refused — here the text is a prefix, so it is reached.
    expect(
      lcsLengthPreparedBounded(preparePattern(pattern, 0, 100), text, 0, 40, 40),
    ).toBe(40)
  })

  // Scoring a window of a longer text is what `partialRatio` does, and an
  // offset start is the one thing the band's row arithmetic never sees from a
  // whole-sequence caller.
  it('is exact over a window of a longer text', () => {
    const rng = makeRng(0xba2d)
    const pattern = randomElements(rng, 70, 4)
    const haystack = randomElements(rng, 400, 4)

    for (let start = 0; start + 70 <= haystack.length; start += 37) {
      const window = Array.from(haystack).slice(start, start + 70)
      const prepared = preparePattern(pattern, 0, pattern.length)

      expect(lcsLengthPrepared(prepared, haystack, start, 70)).toBe(
        lcsReference(pattern, window),
      )
      for (let required = 0; required <= 71; required++) {
        const actual = lcsLengthPreparedBounded(prepared, haystack, start, 70, required)
        const expected = lcsReference(pattern, window)
        if (expected >= required) expect(actual).toBe(expected)
        else expect(actual < 0 || actual === expected).toBe(true)
      }
    }
  })
})

// `partialRatio` and `partialRatioAlignment` do the same search, but only the
// second has to report which window won — so the first is free to visit them in
// the order that prunes best, and does. Only a strictly better window replaces
// the one held, so a different order can pick a different window out of several
// that tie. It must never pick a different score.
describe('the partial-ratio scan orders agree on the score', () => {
  const text = fc.stringMatching(/^[abc ]{0,80}$/)

  it('on inputs of every relative length', () => {
    fc.assert(
      fc.property(text, text, (s1, s2) => {
        expect(partialRatio(s1, s2)).toBe(partialRatioAlignment(s1, s2)?.score ?? 0)
      }),
      { numRuns: 400 },
    )
  })

  it('under a cutoff, which both sides also use to prune', () => {
    fc.assert(
      fc.property(text, text, fc.integer({ min: 0, max: 100 }), (s1, s2, cutoff) => {
        const options = { scoreCutoff: cutoff }
        expect(partialRatio(s1, s2, options)).toBe(
          partialRatioAlignment(s1, s2, options)?.score ?? 0,
        )
      }),
      { numRuns: 400 },
    )
  })

  // Long haystacks are where the two orders differ most: the full-length window
  // search is the divide-and-conquer one, and it now runs before the ~2 * len1
  // shorter windows rather than between them.
  it('on a needle inside a long haystack', () => {
    const needle = 'abcdefghijklmnop'.repeat(4)

    fc.assert(
      fc.property(
        fc.stringMatching(/^[a-p]{0,120}$/),
        fc.stringMatching(/^[a-p]{0,120}$/),
        (head, tail) => {
          const haystack = head + needle + tail
          expect(partialRatio(needle, haystack)).toBe(
            partialRatioAlignment(needle, haystack)?.score ?? 0,
          )
        },
      ),
      { numRuns: 200 },
    )
  })
})

// `levenshteinEditops` takes a hint, and upstream spends it on finding the
// distance first so the alignment can run inside a narrower band. That is a
// second, independent route to the same alignment, so it is exactly the kind of
// shortcut that can disagree with the one it replaces.
describe('an editops hint cannot change the alignment', () => {
  it('agrees with the unhinted alignment whatever the hint', () => {
    fc.assert(
      fc.property(
        fc.stringMatching(/^[abcd]{0,120}$/),
        fc.stringMatching(/^[abcd]{0,120}$/),
        (a, b) => {
          const expected = editopTuples(levenshteinEditops(a, b))

          for (const scoreHint of [0, 1, 5, 31, 32, 64, 200, 100_000]) {
            expect(editopTuples(levenshteinEditops(a, b, { scoreHint }))).toEqual(
              expected,
            )
          }
        },
      ),
      { numRuns: 250 },
    )
  })

  // The hint only pays off when it promises to more than halve the second pass,
  // so a hint at or above half the longest input has to leave the band alone
  // rather than narrow it to something the alignment cannot fit in.
  it('is exact when the hint is far below the real distance', () => {
    const a = 'abcd'.repeat(40)
    const b = 'dcba'.repeat(40)

    expect(editopTuples(levenshteinEditops(a, b, { scoreHint: 0 }))).toEqual(
      editopTuples(levenshteinEditops(a, b)),
    )
  })
})
