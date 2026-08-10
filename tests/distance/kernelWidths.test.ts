// Not ported from RapidFuzz — like `kernels.test.ts`, this guards an
// implementation of ours rather than a behaviour of upstream's.
//
// `kernels.test.ts` checks each kernel against the dynamic program at the
// lengths where one hands over to the next. This file is the other axis: every
// kernel classifies each element of the text it scans into one of four regions
// of the mask table — Latin-1, a window over the pattern's own high symbols, a
// stray slot, or no slot at all — and each kernel carries its own copy of that
// classification, specialised for the pattern width it serves. `pattern.ts`
// says so in as many words: "Any copy that disagrees with this body is a bug."
//
// A copy is therefore only exercised by a pattern of exactly its width scored
// against text holding an element of exactly that region, so the sweep below is
// widths crossed with element kinds, through every scorer that reaches a
// different family of kernels.
import { describe, expect, it } from 'vitest'

import { prepareScorerOf, type PrepareScorer, type Sequence } from '../../src/_common.js'
import { damerauLevenshteinDistance } from '../../src/distance/damerauLevenshtein.js'
import { indelDistance } from '../../src/distance/indel.js'
import { jaroSimilarity } from '../../src/distance/jaro.js'
import { jaroWinklerSimilarity } from '../../src/distance/jaroWinkler.js'
import { lcsSeqDistance, lcsSeqSimilarity } from '../../src/distance/lcsSeq.js'
import {
  levenshteinDistance,
  levenshteinEditops,
} from '../../src/distance/levenshtein.js'
import { osaDistance } from '../../src/distance/osa.js'
import { partialRatio, ratio } from '../../src/fuzz.js'

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
        row[j] + 1,
        row[j - 1] + 1,
        prevDiag + (s1[i - 1] === s2[j - 1] ? 0 : 1),
      )
      prevDiag = above
    }
  }

  return row[s2.length]
}

/** Textbook LCS, O(n*m). */
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

/** Textbook optimal string alignment, O(n*m). */
function osaReference(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const rows = s1.length + 1
  const cols = s2.length + 1
  const d = new Uint32Array(rows * cols)
  for (let i = 0; i < rows; i++) d[i * cols] = i
  for (let j = 0; j < cols; j++) d[j] = j

  for (let i = 1; i < rows; i++) {
    for (let j = 1; j < cols; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1
      let best = Math.min(
        d[(i - 1) * cols + j] + 1,
        d[i * cols + j - 1] + 1,
        d[(i - 1) * cols + j - 1] + cost,
      )
      if (i > 1 && j > 1 && s1[i - 1] === s2[j - 2] && s1[i - 2] === s2[j - 1]) {
        best = Math.min(best, d[(i - 2) * cols + j - 2] + 1)
      }
      d[i * cols + j] = best
    }
  }

  return d[rows * cols - 1]
}

/**
 * Textbook Jaro, O(n*m), with upstream's match window.
 *
 * Two elements match only if they are no further apart than
 * `floor(max(|s1|, |s2|) / 2) - 1`, and half-transpositions are counted over
 * the matched elements in order.
 */
function jaroReference(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  const len1 = s1.length
  const len2 = s2.length
  if (len1 === 0 && len2 === 0) return 1
  if (len1 === 0 || len2 === 0) return 0

  const bound = Math.max(Math.floor(Math.max(len1, len2) / 2) - 1, 0)
  const matched1 = new Array<boolean>(len1).fill(false)
  const matched2 = new Array<boolean>(len2).fill(false)
  let common = 0

  for (let i = 0; i < len1; i++) {
    const low = Math.max(0, i - bound)
    const high = Math.min(len2 - 1, i + bound)
    for (let j = low; j <= high; j++) {
      if (matched2[j] || s1[i] !== s2[j]) continue
      matched1[i] = true
      matched2[j] = true
      common++
      break
    }
  }

  if (common === 0) return 0

  let halfTranspositions = 0
  let k = 0
  for (let i = 0; i < len1; i++) {
    if (!matched1[i]) continue
    while (!matched2[k]) k++
    if (s1[i] !== s2[k]) halfTranspositions++
    k++
  }

  const transpositions = Math.floor(halfTranspositions / 2)
  return (common / len1 + common / len2 + (common - transpositions) / common) / 3
}

/**
 * The four regions of a pattern's mask table, as the elements that reach them.
 *
 * `latin` indexes the Latin-1 block directly. `window` is the contiguous high
 * region a single-script pattern gets. `far` is a second script far enough away
 * that the two together span more than `WINDOW_SPAN_LIMIT` code points, so the
 * pattern is not windowed and every high element takes a stray slot. `opaque`
 * elements are not numbers at all and can only ever be strays; `nothing` is the
 * set that no pattern gives a slot to.
 */
const REGIONS = {
  latin: [97, 98, 99, 100, 101, 102],
  window: [0x410, 0x411, 0x412, 0x413, 0x414, 0x415],
  far: [0x4e00, 0x4e01, 0x4e02, 0x4e03, 0x4e04, 0x4e05],
  // Multi-character strings, not single ones: `convElement` turns `'a'` into
  // `97`, which would put it back in the Latin-1 block.
  opaque: ['ab', 'cd', true, false, null, { tag: 'opaque' }],
  nothing: [Number.NaN, 1.5, -3, 0x20000, 0x30001, Number.NaN],
}

/** Deterministic, so a failure names one case rather than a distribution. */
function generator(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return state / 4294967296
  }
}

function build(palette: readonly unknown[], length: number, seed: number): unknown[] {
  const next = generator(seed)
  const out = new Array<unknown>(length)
  for (let i = 0; i < length; i++) out[i] = palette[Math.floor(next() * palette.length)]
  return out
}

/** One palette per pattern shape, and per text shape scored against it. */
const PALETTES: ReadonlyMap<string, readonly unknown[]> = new Map([
  ['latin', REGIONS.latin],
  ['windowed', [...REGIONS.latin, ...REGIONS.window]],
  ['strays', [...REGIONS.window, ...REGIONS.far]],
  ['opaque', [...REGIONS.latin, ...REGIONS.opaque]],
  [
    'every region',
    [...REGIONS.latin, ...REGIONS.window, ...REGIONS.far, ...REGIONS.opaque],
  ],
  ['unmatched', [...REGIONS.latin, ...REGIONS.window, ...REGIONS.nothing]],
])

// One word, two, three, four and five or more — the widths each kernel family
// splits on — plus the boundaries between them and the first length past a
// four-word held pattern, which is the only way into the wide prepared kernels.
const WIDTHS = [1, 2, 5, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 160, 200]

/**
 * The same regions again, spelled as characters.
 *
 * Every kernel reads a string with `charCodeAt` and an array by index, and the
 * two are separate branches with separate copies of the classification below
 * them — a BMP-only pair stays a pair of strings all the way down, so an
 * array-only sweep never enters the string half of any of them. Only the
 * regions a `char` can reach are here: `opaque` has no spelling as text.
 */
const TEXT_PALETTES: ReadonlyMap<string, string> = new Map([
  ['latin text', 'abcdef'],
  ['windowed text', 'abcАБВ'],
  ['stray text', 'АБВ一丁丂'],
  ['every text region', 'abАБ一丁'],
])

interface Pair {
  readonly what: string
  readonly s1: Sequence
  readonly s2: Sequence
}

/** Every width crossed with every palette, both same-length and not. */
function* pairs(): Generator<Pair> {
  let seed = 1
  for (const width of WIDTHS) {
    for (const [name, palette] of PALETTES) {
      for (const [otherName, other] of PALETTES) {
        seed++
        if ((seed & 3) !== 0 && name !== otherName) continue
        yield {
          what: `${width} ${name} against ${otherName}`,
          s1: build(palette, width, seed),
          s2: build(other, width, seed + 7919),
        }
        yield {
          what: `${width} ${name} against a longer ${otherName}`,
          s1: build(palette, width, seed + 104_729),
          s2: build(other, width + 37, seed + 15_485_863),
        }
      }
    }

    for (const [name, palette] of TEXT_PALETTES) {
      for (const [otherName, other] of TEXT_PALETTES) {
        seed++
        if ((seed & 3) !== 0 && name !== otherName) continue
        yield {
          what: `${width} ${name} against ${otherName}`,
          s1: text(palette, width, seed),
          s2: text(other, width, seed + 7919),
        }
        yield {
          what: `${width} ${name} against a longer ${otherName}`,
          s1: text(palette, width, seed + 104_729),
          s2: text(other, width + 37, seed + 15_485_863),
        }
      }
    }
  }
}

function text(palette: string, length: number, seed: number): string {
  return build([...palette], length, seed).join('')
}

const PAIRS = [...pairs()]

function preparedOf(scorer: object): PrepareScorer {
  const factory = prepareScorerOf(scorer)
  if (factory === null) throw new Error('scorer has no prepared factory')
  return factory
}

describe('every mask region, at every pattern width', () => {
  it('scores Levenshtein as the dynamic program does', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const expected = levenshteinReference(s1, s2)
      expect(levenshteinDistance(s1, s2), what).toBe(expected)
      expect(levenshteinDistance(s2, s1), what).toBe(expected)
    }
  })

  // The prepared kernels are a second set, split by width the same way and
  // reached only by holding the pattern across candidates.
  it('scores a held Levenshtein pattern the same', () => {
    const prepare = preparedOf(levenshteinDistance)
    for (const { what, s1, s2 } of PAIRS) {
      const expected = levenshteinReference(s1, s2)
      expect(prepare(s1, {})(s2, null, null), what).toBe(expected)
      expect(prepare(s2, {})(s1, null, null), what).toBe(expected)
    }
  })

  it('scores LCS as the dynamic program does', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const expected = lcsReference(s1, s2)
      expect(lcsSeqSimilarity(s1, s2), what).toBe(expected)
      expect(lcsSeqDistance(s1, s2), what).toBe(Math.max(s1.length, s2.length) - expected)
      expect(indelDistance(s1, s2), what).toBe(s1.length + s2.length - 2 * expected)
    }
  })

  it('scores a held LCS pattern the same', () => {
    const prepare = preparedOf(lcsSeqSimilarity)
    for (const { what, s1, s2 } of PAIRS) {
      const expected = lcsReference(s1, s2)
      expect(prepare(s1, {})(s2, null, null), what).toBe(expected)
      expect(prepare(s2, {})(s1, null, null), what).toBe(expected)
    }
  })

  it('scores OSA and Damerau-Levenshtein as their dynamic programs do', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const expected = osaReference(s1, s2)
      expect(osaDistance(s1, s2), what).toBe(expected)
      expect(osaDistance(s2, s1), what).toBe(expected)
      // Damerau-Levenshtein allows any transposition, so it can only be lower.
      expect(damerauLevenshteinDistance(s1, s2), what).toBeLessThanOrEqual(expected)
    }
  })

  // `ratio` and `partialRatio` reach the same LCS kernels through a normalised
  // score and, for the partial one, through a window scan over the longer side.
  it('scores the fuzz ratios consistently with LCS', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const maximum = s1.length + s2.length
      const expected = (1 - (maximum - 2 * lcsReference(s1, s2)) / maximum) * 100
      expect(ratio(s1, s2), what).toBeCloseTo(expected, 9)
      // The window scan is its own alignment, so it is not bounded by `ratio`;
      // what it must be is symmetric and inside the range a score can take.
      const partial = partialRatio(s1, s2)
      expect(partial, what).toBeGreaterThanOrEqual(0)
      expect(partial, what).toBeLessThanOrEqual(100)
      expect(partialRatio(s2, s1), what).toBeCloseTo(partial, 9)
    }
  })

  // Jaro's kernels split by width too, and its match window means the mask a
  // symbol reaches is read once per word the window spans rather than once.
  it('scores Jaro as the dynamic program does', () => {
    const prepare = preparedOf(jaroSimilarity)
    for (const { what, s1, s2 } of PAIRS) {
      const expected = jaroReference(s1, s2)
      expect(jaroSimilarity(s1, s2), what).toBeCloseTo(expected, 12)
      expect(jaroSimilarity(s2, s1), what).toBeCloseTo(expected, 12)
      expect(prepare(s1, {})(s2, null, null), what).toBeCloseTo(expected, 12)
      expect(prepare(s2, {})(s1, null, null), what).toBeCloseTo(expected, 12)
    }
  })

  it('scores Jaro-Winkler consistently with Jaro', () => {
    const prepare = preparedOf(jaroWinklerSimilarity)
    for (const { what, s1, s2 } of PAIRS) {
      const scored = jaroWinklerSimilarity(s1, s2)
      expect(scored, what).toBeGreaterThanOrEqual(jaroReference(s1, s2) - 1e-12)
      expect(prepare(s1, {})(s2, null, null), what).toBeCloseTo(scored, 12)
      expect(prepare(s2, {})(s1, null, null), what).toBeCloseTo(scored, 12)
    }
  })

  it('scores a held OSA pattern the same', () => {
    const prepare = preparedOf(osaDistance)
    for (const { what, s1, s2 } of PAIRS) {
      const expected = osaReference(s1, s2)
      expect(prepare(s1, {})(s2, null, null), what).toBe(expected)
      expect(prepare(s2, {})(s1, null, null), what).toBe(expected)
    }
  })

  it('scores a held ratio pattern the same', () => {
    const prepare = preparedOf(ratio)
    const preparePartial = preparedOf(partialRatio)
    for (const { what, s1, s2 } of PAIRS) {
      expect(prepare(s1, {})(s2, null, null), what).toBeCloseTo(ratio(s1, s2), 9)
      expect(preparePartial(s1, {})(s2, null, null), what).toBeCloseTo(
        partialRatio(s1, s2),
        9,
      )
    }
  })
})

// A cutoff sends the same pairs through the banded kernels instead, which hold
// a diagonal band rather than the whole row and have their own copy of the
// element classification again.
describe('every mask region, under a cutoff', () => {
  it('returns the exact distance inside the bound and a sentinel outside', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const exact = levenshteinReference(s1, s2)
      for (const cutoff of [0, 1, 3, 4, 15, 16, 40, exact - 1, exact, exact + 1]) {
        if (cutoff < 0) continue
        const scored = levenshteinDistance(s1, s2, { scoreCutoff: cutoff })
        expect(scored, `${what} at ${cutoff}`).toBe(exact <= cutoff ? exact : cutoff + 1)
      }
    }
  })

  it('honours a hint without changing the answer', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const exact = levenshteinReference(s1, s2)
      for (const hint of [0, 4, 16, 64, exact]) {
        expect(
          levenshteinDistance(s1, s2, { scoreHint: hint }),
          `${what} at ${hint}`,
        ).toBe(exact)
      }
    }
  })

  // Indel and LCS have a banded kernel of their own, with its own copy of the
  // classification, and this reaches it through the direct entry points rather
  // than a held pattern. Deliberately not through the prepared path: which of
  // the two kernels that picks is a dispatch decision — `sharesAffix` moved
  // these very pairs onto the held pattern and took this copy's coverage with
  // them — and a classification every kernel carries separately has to be
  // covered whichever way the dispatch happens to go.
  it('scores Indel and LCS exactly inside the bound', () => {
    for (const { what, s1, s2 } of PAIRS) {
      const lcs = lcsReference(s1, s2)
      const indel = s1.length + s2.length - 2 * lcs
      for (const cutoff of [0, 1, 3, 8, 32, indel - 1, indel, indel + 1]) {
        if (cutoff < 0) continue
        expect(
          indelDistance(s1, s2, { scoreCutoff: cutoff }),
          `${what} at ${cutoff}`,
        ).toBe(indel <= cutoff ? indel : cutoff + 1)
      }
      for (const cutoff of [0, 1, 8, lcs, lcs + 1]) {
        expect(
          lcsSeqSimilarity(s1, s2, { scoreCutoff: cutoff }),
          `${what} at ${cutoff}`,
        ).toBe(lcs >= cutoff ? lcs : 0)
      }
    }
  })
})

// A held pattern under a cutoff is a third set again: the bounded and banded
// kernels hold a window of the row rather than all of it, and each carries its
// own copy of the classification.
describe('every mask region, with a held pattern under a cutoff', () => {
  it('scores Levenshtein exactly inside the bound', () => {
    const prepare = preparedOf(levenshteinDistance)
    for (const { what, s1, s2 } of PAIRS) {
      const exact = levenshteinReference(s1, s2)
      const score = prepare(s1, {})
      for (const cutoff of [0, 1, 3, 4, 15, 16, 40, exact, exact + 1]) {
        expect(score(s2, cutoff, null), `${what} at ${cutoff}`).toBe(
          exact <= cutoff ? exact : cutoff + 1,
        )
      }
    }
  })

  it('scores Indel and LCS exactly inside the bound', () => {
    const prepareIndel = preparedOf(indelDistance)
    const prepareLcs = preparedOf(lcsSeqSimilarity)
    for (const { what, s1, s2 } of PAIRS) {
      const lcs = lcsReference(s1, s2)
      const indel = s1.length + s2.length - 2 * lcs
      const scoreIndel = prepareIndel(s1, {})
      const scoreLcs = prepareLcs(s1, {})

      for (const cutoff of [0, 1, 3, 8, 32, indel, indel + 1]) {
        expect(scoreIndel(s2, cutoff, null), `${what} at ${cutoff}`).toBe(
          indel <= cutoff ? indel : cutoff + 1,
        )
      }
      for (const cutoff of [0, 1, 8, lcs, lcs + 1]) {
        expect(scoreLcs(s2, cutoff, null), `${what} at ${cutoff}`).toBe(
          lcs >= cutoff ? lcs : 0,
        )
      }
    }
  })

  // `ratio` takes the bounded held kernel only above a cutoff of 70 and a
  // combined length of 128, which no unbounded sweep reaches.
  it('scores a held ratio exactly inside the bound', () => {
    const prepare = preparedOf(ratio)
    for (const { what, s1, s2 } of PAIRS) {
      const exact = ratio(s1, s2)
      const score = prepare(s1, {})
      for (const cutoff of [0, 25, 70, 90, 100]) {
        expect(score(s2, cutoff, null), `${what} at ${cutoff}`).toBeCloseTo(
          exact >= cutoff ? exact : 0,
          9,
        )
      }
    }
  })
})

// `editops` walks the recurrence back, which needs every row rather than the
// last vector, and so runs a second implementation of the same recurrence.
describe('editops over every mask region', () => {
  it('produces a script whose length is the distance', () => {
    for (const { what, s1, s2 } of PAIRS) {
      if (s1.length > 70) continue
      const ops = levenshteinEditops(s1, s2)
      expect(ops.operations.length, what).toBe(levenshteinReference(s1, s2))
      expect(ops.srcLen, what).toBe(s1.length)
      expect(ops.destLen, what).toBe(s2.length)
    }
  })
})
