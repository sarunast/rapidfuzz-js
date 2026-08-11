/**
 * Shared Myers bit-parallel Levenshtein, over 32-bit words.
 *
 * The whole family lives in one module — single-word, multi-word, banded and
 * small-band kernels, the mbleven models and the prepared-mask readers —
 * because {@link levenshteinUniform}'s dispatch is only legible beside the
 * kernels it chooses between.
 *
 * {@link shiftedPatternMatches} is here rather than in `pattern.ts` despite
 * reading a `PatternMask`: {@link levenshteinSmallBand} is its only caller, and
 * the shifted window it produces is part of that strategy rather than something
 * a prepared pattern generally offers.
 */

import {
  affixLen1,
  affixLen2,
  affixPrefix,
  directSlots,
  directStamps,
  bandVector,
  blockMasksFor,
  buildWordMasks,
  clearRange,
  directLimit,
  maskPoolOf,
  measureAffix,
  rowVector,
  rowVectorN,
  wideSlots,
} from '../../shared/bitmask/blockMasks.js'
import { preparePattern, type PatternMask } from '../../shared/bitmask/pattern.js'

// Declared here rather than imported from `shared.ts`, which holds the
// canonical definitions and documents the invariant they keep. These are read
// once per element of the streamed input, where a cross-module binding does not
// fold the way a module-local `const` does — measured at +3% on Latin-1 and
// +15% on Cyrillic for a loop of this shape. Any copy that disagrees with
// `shared.ts` is a bug.
const WORD_BITS = 32
const WORD_SHIFT = 5
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

/**
 * Myers' bit-parallel Levenshtein over one word, for uniform weights.
 *
 * Same recurrence as `levenshteinMatrix` in `_bitParallel.ts`, with the row
 * history dropped. Bits above `patternLength` in the word are left set; they
 * describe pattern positions that do not exist, and since carries only travel
 * upward they can never influence the score read off bit `patternLength - 1`.
 */
function levenshteinOneWord(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const stamp = buildWordMasks(pattern, patternStart, patternLength)
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const top = 1 << (patternLength - 1)

  let vp = -1
  let vn = 0
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    let x: number

    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      x = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < limit &&
      (symbol | 0) === symbol
    ) {
      x = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (symbol === symbol) {
      x = wide.get(symbol) ?? 0
    } else {
      x = 0
    }

    // The addition may exceed 32 bits; `^` coerces back to int32, which
    // discards the carry-out exactly as the algorithm requires.
    const d0 = (((x & vp) + vp) ^ vp) | x | vn | 0
    let hp = vn | ~(d0 | vp)
    let hn = d0 & vp

    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--

    hp = (hp << 1) | 1
    hn = hn << 1
    vp = hn | ~(d0 | hp)
    vn = hp & d0
  }

  return distance
}

/**
 * Myers' Levenshtein carried across words, with the carry chains fused.
 *
 * Nothing but the dispatch: the four kernels below run the same recurrence, and
 * differ only in where the row lives.
 *
 * ## Why the narrow widths are written out
 *
 * {@link levenshteinWideWords} keeps the two row vectors in the shared typed
 * arrays, so every word of every text element is a pair of loads and a pair of
 * stores. Spelling a width out lets the whole row stay in locals for the length
 * of the scan, and the loop counter, its bounds check and the last-word test go
 * with it — measured at 1.5x for two, three and four words, against text of
 * either 512 or 4096 elements.
 *
 * The same trade as `lcsLengthPrepared`'s two- and three-word cases, one width
 * further. Note whose inputs those are, because it is not the same set: nothing
 * in the fuzz family reaches this kernel — every scorer there is built on
 * `lcsSeqLength*`, and `dist/fuzz.js` does not import this module at all. What
 * arrives here is a Levenshtein scorer called directly, `levenshteinEditops`,
 * or an `extract`/`scoreMatrix` given one, so 33 to 128 elements is the band of
 * names, identifiers and short fields those are handed.
 *
 * It keeps paying past that — a five-word kernel measured 1.56x, six 1.37x and
 * eight 1.23x — so the stopping point is about how much near-duplicate code a
 * width is worth, not about where the win ends.
 *
 * ## What is shared, and what cannot be
 *
 * The per-element table lookup is {@link patternOffset}, called rather than
 * written out at each width — it hands back one number, and extracting it
 * measured level over three runs at every width from two words to eight. The
 * warning in `shared.ts` about this test being written out at each site is
 * about the *mask*-returning form in {@link levenshteinOneWord}, where `0` is
 * both "absent" and a legitimate mask, and about the LCS kernels that predate
 * this one.
 *
 * The per-word recurrence cannot follow it. Each word step produces five values
 * the next word reads — the two row words, the addition's carry, and the two
 * shifted-delta carries — and JavaScript returns one thing. A helper would have
 * to hand back an object or write through a scratch array, which is the memory
 * traffic these kernels exist to avoid. So the recurrence stays spelled out at
 * each width and the lookup above it does not.
 */
/**
 * Where one element's masks start in the shared pool, or `-1` if the pattern
 * does not hold it.
 *
 * Every caller hoists the four tables itself and passes them in, rather than
 * this reaching for the imported bindings: a live binding read per element is
 * what `shared.ts` warns costs 3 to 17 percent, and passing them keeps the
 * hoist visible at the call site. Reading them here instead was measured too,
 * and lands about a percent behind.
 */
function patternOffset(
  symbol: unknown,
  stamp: number,
  slots: Int32Array,
  stamps: Int32Array,
  wide: Map<unknown, number>,
  limit: number,
): number {
  if (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  ) {
    return stamps[symbol] === stamp ? slots[symbol] : -1
  }
  if (
    typeof symbol === 'number' &&
    symbol >= DIRECT_LOOKUP_LIMIT &&
    symbol < limit &&
    (symbol | 0) === symbol
  ) {
    return stamps[symbol] === stamp ? slots[symbol] : -1
  }
  if (symbol === symbol) return wide.get(symbol) ?? -1
  return -1
}

function levenshteinManyWords(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const words = (patternLength + WORD_BITS - 1) >>> WORD_SHIFT
  // Building the masks can widen the shared table, so every kernel below hoists
  // `directLimit` and the buffers after this call rather than before it.
  const stamp = blockMasksFor(pattern, patternStart, patternLength, words)

  if (words === 2) {
    return levenshteinTwoWords(patternLength, text, textStart, textLength, stamp)
  }
  if (words === 3) {
    return levenshteinThreeWords(patternLength, text, textStart, textLength, stamp)
  }
  if (words === 4) {
    return levenshteinFourWords(patternLength, text, textStart, textLength, stamp)
  }
  return levenshteinWideWords(patternLength, words, text, textStart, textLength, stamp)
}

/**
 * Myers' Levenshtein over a two-word pattern, the row held in locals.
 *
 * The recurrence is {@link levenshteinManyWords}'s, with `w` resolved: word 0
 * opens the carry chains, and the last word closes them and is the only one that
 * moves the distance. See that function for why this is written out.
 */
function levenshteinTwoWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  stamp: number,
): number {
  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  let vp0 = -1
  let vp1 = -1
  let vn0 = 0
  let vn1 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    const x0 = offset < 0 ? 0 : pool[offset]
    const x1 = offset < 0 ? 0 : pool[offset + 1]

    // Modular addition with the carry recovered by bit arithmetic, so no value
    // in this loop leaves the small-integer range. See the matching comment in
    // `lcsManyWords`.
    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    const addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    const carryP = hp >>> 31
    const carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0
  }

  return distance
}

/** Three words of pattern, the row held in locals — see {@link levenshteinTwoWords}. */
function levenshteinThreeWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  stamp: number,
): number {
  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  let vp0 = -1
  let vp1 = -1
  let vp2 = -1
  let vn0 = 0
  let vn1 = 0
  let vn2 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    const x0 = offset < 0 ? 0 : pool[offset]
    const x1 = offset < 0 ? 0 : pool[offset + 1]
    const x2 = offset < 0 ? 0 : pool[offset + 2]

    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    let addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    let carryP = hp >>> 31
    let carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    addCarry = ((addend & vp1) | ((addend | vp1) & ~sum)) >>> 31
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0

    addend = x2 & vp2
    sum = (addend + vp2 + addCarry) | 0
    d0 = (sum ^ vp2) | x2 | vn2 | 0
    hp = vn2 | ~(d0 | vp2)
    hn = d0 & vp2
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp2 = shiftedN | ~(d0 | shiftedP)
    vn2 = shiftedP & d0
  }

  return distance
}

/** Four words of pattern, the row held in locals — see {@link levenshteinTwoWords}. */
function levenshteinFourWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  stamp: number,
): number {
  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  let vp0 = -1
  let vp1 = -1
  let vp2 = -1
  let vp3 = -1
  let vn0 = 0
  let vn1 = 0
  let vn2 = 0
  let vn3 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    const x0 = offset < 0 ? 0 : pool[offset]
    const x1 = offset < 0 ? 0 : pool[offset + 1]
    const x2 = offset < 0 ? 0 : pool[offset + 2]
    const x3 = offset < 0 ? 0 : pool[offset + 3]

    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    let addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    let carryP = hp >>> 31
    let carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    addCarry = ((addend & vp1) | ((addend | vp1) & ~sum)) >>> 31
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0

    addend = x2 & vp2
    sum = (addend + vp2 + addCarry) | 0
    addCarry = ((addend & vp2) | ((addend | vp2) & ~sum)) >>> 31
    d0 = (sum ^ vp2) | x2 | vn2 | 0
    hp = vn2 | ~(d0 | vp2)
    hn = d0 & vp2
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp2 = shiftedN | ~(d0 | shiftedP)
    vn2 = shiftedP & d0

    addend = x3 & vp3
    sum = (addend + vp3 + addCarry) | 0
    d0 = (sum ^ vp3) | x3 | vn3 | 0
    hp = vn3 | ~(d0 | vp3)
    hn = d0 & vp3
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp3 = shiftedN | ~(d0 | shiftedP)
    vn3 = shiftedP & d0
  }

  return distance
}

/**
 * Myers' Levenshtein over five words of pattern or more, the row in the shared
 * vectors.
 *
 * Two things the loop no longer does per word. The last word is peeled out, so
 * no word tests whether it is the one that moves the distance; and an element
 * the pattern does not hold takes a loop of its own, so no word asks. Neither
 * can be hoisted in the narrow kernels above — there is no loop left to hoist
 * out of — and together they measured about 1.12x here, over five to eight
 * words. The absent-element loop is the same recurrence with `x` at zero
 * folded through it; unlike the LCS row it cannot be skipped, since `d0` still
 * picks up `vn` and the horizontal deltas still shift.
 */
function levenshteinWideWords(
  patternLength: number,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  stamp: number,
): number {
  const vp = rowVector(words)
  const vn = rowVectorN(words)
  clearRange(vp, -1, 0, words)
  clearRange(vn, 0, 0, words)

  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const lastWord = words - 1
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    let addCarry = 0
    let carryP = 1
    let carryN = 0

    if (offset < 0) {
      for (let w = 0; w < lastWord; w++) {
        const vpWord = vp[w]
        const vnWord = vn[w]

        const sum = (vpWord + addCarry) | 0
        addCarry = (vpWord & ~sum) >>> 31

        const d0 = (sum ^ vpWord) | vnWord | 0
        const hp = vnWord | ~(d0 | vpWord)
        const hn = d0 & vpWord

        const shiftedP = (hp << 1) | carryP
        const shiftedN = (hn << 1) | carryN
        carryP = hp >>> 31
        carryN = hn >>> 31

        vp[w] = shiftedN | ~(d0 | shiftedP)
        vn[w] = shiftedP & d0
      }

      const vpWord = vp[lastWord]
      const vnWord = vn[lastWord]

      const sum = (vpWord + addCarry) | 0

      const d0 = (sum ^ vpWord) | vnWord | 0
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      // No `distance--` beside it: with no match bits, `d0` is `vn` and `hn` is
      // `vn & vp`, and a cell is never both ahead and behind — so the negative
      // delta this word could carry is identically zero.
      if ((hp & top) !== 0) distance++

      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN

      vp[lastWord] = shiftedN | ~(d0 | shiftedP)
      vn[lastWord] = shiftedP & d0
      continue
    }

    for (let w = 0; w < lastWord; w++) {
      const vpWord = vp[w]
      const vnWord = vn[w]
      const x = pool[offset + w]

      // Modular addition with the carry recovered by bit arithmetic, so no
      // value in this loop leaves the small-integer range. See the matching
      // comment in `lcsManyWords`.
      const addend = x & vpWord
      const sum = (addend + vpWord + addCarry) | 0
      addCarry = ((addend & vpWord) | ((addend | vpWord) & ~sum)) >>> 31

      const d0 = (sum ^ vpWord) | x | vnWord | 0
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN
      carryP = hp >>> 31
      carryN = hn >>> 31

      vp[w] = shiftedN | ~(d0 | shiftedP)
      vn[w] = shiftedP & d0
    }

    const vpWord = vp[lastWord]
    const vnWord = vn[lastWord]
    const x = pool[offset + lastWord]

    const addend = x & vpWord
    const sum = (addend + vpWord + addCarry) | 0

    const d0 = (sum ^ vpWord) | x | vnWord | 0
    const hp = vnWord | ~(d0 | vpWord)
    const hn = d0 & vpWord

    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--

    const shiftedP = (hp << 1) | carryP
    const shiftedN = (hn << 1) | carryN

    vp[lastWord] = shiftedN | ~(d0 | shiftedP)
    vn[lastWord] = shiftedP & d0
  }

  return distance
}

/**
 * Ukkonen-banded blocked Myers — port of `levenshtein_hyrroe2003_block`.
 *
 * {@link levenshteinManyWords} walks every word of every row, which is the best
 * anyone can do without a budget. Given one, only the words the band can still
 * reach matter: `firstWord`/`lastWord` track that band, the budget is tightened
 * after each row from the score at the band's far edge, and once the band closes
 * the answer is known to be out of reach and the scan stops.
 *
 * The recurrence differs from {@link levenshteinManyWords}: there, the addition
 * carries between words, so every word of a row has to run. Here the horizontal
 * delta crosses the word boundary through `hnCarry` instead — which is what lets
 * a row start part-way along, at `firstWord`.
 */
function levenshteinManyWordsBanded(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  budget: number,
): number {
  const words = (patternLength + WORD_MASK) >>> WORD_SHIFT
  const stamp = blockMasksFor(pattern, patternStart, patternLength, words)

  const vp = rowVector(words)
  const vn = rowVectorN(words)
  const scores = bandVector(words)

  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const last = 1 << ((patternLength - 1) & WORD_MASK)
  const stringText = typeof text === 'string'

  // The final row of a word, as an index into the pattern. Every band test
  // below compares that row against the diagonal the answer has to reach.
  const rowNumberOf = (word: number): number =>
    word + 1 === words ? patternLength - 1 : (word + 1) * WORD_BITS - 1

  let max = Math.min(budget, Math.max(patternLength, textLength))
  let firstWord = 0
  let lastWord =
    Math.min(
      words,
      Math.ceil(
        (Math.min(max, Math.floor((max + patternLength - textLength) / 2)) + 1) /
          WORD_BITS,
      ),
    ) - 1

  // Only the words the band starts on are set up. A tight budget on a long
  // pattern opens on one word out of a hundred and more, and the two vectors
  // plus the score row were being initialised across all of them.
  //
  // Safe because the band only ever widens through the branch below, which
  // gives the word it steps onto its own `vp`, `vn` and `scores` entry before
  // reading any of them — and because a word past the band is never read: the
  // answer lives in `scores[words - 1]`, which the `lastWord < words - 1` test
  // at the end refuses to read unless the band actually arrived there.
  // `firstWord` only advances, so a word the band has left is never revisited.
  const opening = lastWord + 1
  clearRange(vp, -1, 0, opening)
  clearRange(vn, 0, 0, opening)
  for (let i = 0; i < opening; i++) {
    scores[i] = i + 1 === words ? patternLength : (i + 1) * WORD_BITS
  }

  // Carries out of the word the row is currently at, read by the band
  // adjustment below as well as by the next word.
  let carryP = 1
  let carryN = 0
  const limit = directLimit

  for (let row = 0; row < textLength; row++) {
    const symbol = stringText ? text.charCodeAt(textStart + row) : text[textStart + row]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    // The recurrence below is written out twice — here and for the widening
    // word further down — rather than shared as a helper. It was a closure, and
    // a closure that assigns `carryP`/`carryN` forces both into a heap context
    // cell for the whole call, so every read of them (including the ones outside
    // it) became a context load instead of a register. One allocation per text
    // element on top of that.
    carryP = 1
    carryN = 0
    for (let word = firstWord; word <= lastWord; word++) {
      const matches = offset < 0 ? 0 : pool[offset + word]
      const vpWord = vp[word]
      const vnWord = vn[word]

      const x = matches | carryN
      // The overflow out of this addition is dropped rather than carried on,
      // which is exactly what the 64-bit original does. `| 0` reproduces the
      // wrap at the word width.
      const d0 = ((((x & vpWord) + vpWord) | 0) ^ vpWord) | x | vnWord
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      const carriedP = carryP
      const carriedN = carryN
      if (word < words - 1) {
        carryP = hp >>> 31
        carryN = hn >>> 31
      } else {
        carryP = (hp & last) !== 0 ? 1 : 0
        carryN = (hn & last) !== 0 ? 1 : 0
      }

      const shiftedP = (hp << 1) | carriedP
      const shiftedN = (hn << 1) | carriedN
      vp[word] = shiftedN | ~(d0 | shiftedP)
      vn[word] = shiftedP & d0

      scores[word] += carryP - carryN
    }

    max = Math.min(
      max,
      scores[lastWord] +
        Math.max(textLength - row - 1, patternLength - (1 + lastWord) * WORD_BITS + 2),
    )

    // Widen by at most one word: anything beyond the next one is certainly
    // still beneath the band.
    if (lastWord + 1 < words) {
      const reach =
        max + 2 * WORD_BITS + row + patternLength - (scores[lastWord] + 2 + textLength)
      if (rowNumberOf(lastWord) < reach) {
        lastWord++
        vp[lastWord] = -1
        vn[lastWord] = 0
        const held =
          lastWord + 1 === words ? ((patternLength - 1) & WORD_MASK) + 1 : WORD_BITS
        scores[lastWord] = scores[lastWord - 1] + held - carryP + carryN

        // The second copy of the recurrence above, for the word just widened
        // into. `word` is `lastWord`, so the final-word test is spelled against
        // it directly.
        const matches = offset < 0 ? 0 : pool[offset + lastWord]
        const vpWord = vp[lastWord]
        const vnWord = vn[lastWord]

        const x = matches | carryN
        const d0 = ((((x & vpWord) + vpWord) | 0) ^ vpWord) | x | vnWord
        // `vp` was set to all ones two statements ago, so `~(d0 | vpWord)` is
        // zero and with it `hp`: a word the band has only now reached has no
        // positive horizontal delta to carry out of, whichever word it is. That
        // leaves `hn` as `d0` and the shifted positive vector as the carry in.
        const hn = d0

        const carriedP = carryP
        const carriedN = carryN
        carryP = 0
        carryN = lastWord < words - 1 ? hn >>> 31 : (hn & last) !== 0 ? 1 : 0

        const shiftedN = (hn << 1) | carriedN
        vp[lastWord] = shiftedN | ~(d0 | carriedP)
        vn[lastWord] = carriedP & d0

        scores[lastWord] += carryP - carryN
      }
    }

    while (lastWord >= firstWord) {
      // A word is in band while its score can still reach the budget, and
      // while its rows have not fallen off the diagonal the answer sits on.
      // Checking the block's first cell settles it for every cell in the block.
      const reach =
        max +
        2 * WORD_BITS +
        row +
        patternLength +
        1 -
        (scores[lastWord] + 2 + textLength)
      if (scores[lastWord] < max + WORD_BITS && rowNumberOf(lastWord) <= reach) break
      lastWord--
    }

    while (firstWord <= lastWord) {
      const reach = scores[firstWord] + patternLength + row - (max + textLength)
      if (scores[firstWord] < max + WORD_BITS && rowNumberOf(firstWord) >= reach) break
      firstWord++
    }

    // The band has closed: every remaining cell is already past the budget.
    if (lastWord < firstWord) return budget + 1
  }

  // The answer sits in the last word, and the loop above has already returned
  // `budget + 1` for every run that could not reach it or could not stay inside
  // the budget: the band closes from both ends, and it closes before the last
  // row when either would have happened.
  return scores[words - 1]
}

/** Upstream's encoded Levenshtein mbleven models for budgets below four. */
const LEVENSHTEIN_MBLEVEN_OPS: ReadonlyArray<readonly number[]> = [
  [0x03],
  [0x01],
  [0x0f, 0x09, 0x06],
  [0x0d, 0x07],
  [0x05],
  [0x3f, 0x27, 0x2d, 0x39, 0x36, 0x1e, 0x1b],
  [0x3d, 0x37, 0x1f, 0x25, 0x19, 0x16],
  [0x35, 0x1d, 0x17],
  [0x15],
]

function levenshteinMbleven(
  first: ArrayLike<unknown>,
  firstStart: number,
  firstLength: number,
  second: ArrayLike<unknown>,
  secondStart: number,
  secondLength: number,
  budget: number,
): number {
  if (firstLength < secondLength) {
    return levenshteinMbleven(
      second,
      secondStart,
      secondLength,
      first,
      firstStart,
      firstLength,
      budget,
    )
  }

  const lengthDifference = firstLength - secondLength
  if (budget === 0) return 1
  // A budget of one never succeeds here. `bounded` is only called with a budget
  // below the longer trimmed input, so `firstLength` is at least two — and two
  // sequences whose ends have already been trimmed apart cannot be one edit
  // from each other unless one of them is a single element.
  if (budget === 1) return 2

  const scripts =
    LEVENSHTEIN_MBLEVEN_OPS[(budget + budget * budget) / 2 + lengthDifference - 1]
  let best = budget + 1

  for (let model = 0; model < scripts.length; model++) {
    let operations = scripts[model]
    let i = 0
    let j = 0
    let distance = 0

    while (i < firstLength && j < secondLength) {
      if (first[firstStart + i] !== second[secondStart + j]) {
        distance++
        if (operations === 0) break
        if ((operations & 1) !== 0) i++
        if ((operations & 2) !== 0) j++
        operations >>>= 2
      } else {
        i++
        j++
      }
    }

    distance += firstLength - i + (secondLength - j)
    if (distance < best) best = distance
  }

  return best <= budget ? best : budget + 1
}

/** Fill Levenshtein distances for every prefix of a prepared pattern. */
export function levenshteinPreparedRow(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  textStep: number,
  out: Uint32Array,
): void {
  // No bounds test and no empty-pattern case: `out` is sized from
  // `prepared.length` by the only caller, which never prepares an empty pattern.
  const words = prepared.words
  const vp = rowVector(words)
  const vn = rowVectorN(words)
  clearRange(vp, -1, 0, words)
  clearRange(vn, 0, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let i = 0; i < textLength; i++) {
    const index = textStart + i * textStep
    const symbol = stringText ? text.charCodeAt(index) : text[index]
    // Where this element's masks start is settled once per element, as in the
    // LCS kernels — inside the word loop a four-word pattern answered the same
    // question four times, one of them a `Map` lookup.
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    let addCarry = 0
    let carryP = 1
    let carryN = 0
    for (let word = 0; word < words; word++) {
      const vpWord = vp[word]
      const vnWord = vn[word]
      const matches = base < 0 ? 0 : masks[base + word]
      const addend = matches & vpWord
      const sum = (addend + vpWord + addCarry) | 0
      addCarry = ((addend & vpWord) | ((addend | vpWord) & ~sum)) >>> 31
      const d0 = (sum ^ vpWord) | matches | vnWord | 0
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord
      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN
      carryP = hp >>> 31
      carryN = hn >>> 31
      vp[word] = shiftedN | ~(d0 | shiftedP)
      vn[word] = shiftedP & d0
    }
  }

  out[0] = textLength
  for (let i = 0; i < prepared.length; i++) {
    const bit = 1 << (i & WORD_MASK)
    const word = i >>> WORD_SHIFT
    out[i + 1] =
      out[i] + ((vp[word] & bit) !== 0 ? 1 : 0) - ((vn[word] & bit) !== 0 ? 1 : 0)
  }
}

/**
 * Myers' Levenshtein of the held pattern against `text[textStart, +textLength)`.
 *
 * The recurrences are those of {@link levenshteinOneWord} and
 * {@link levenshteinManyWords}; only where the match masks come from differs.
 * Rebuilding them costs `O(|pattern|)` writes plus a `Map` clear per call, which
 * is the whole of the work when a short pattern is scored against many texts —
 * `scoreMatrix` and collection search do exactly that.
 *
 * No common affix is trimmed, for the reason given on {@link lcsLengthPrepared}:
 * trimming shortens the pattern by a different amount for every text, which is
 * what the held masks cannot express. Callers gate on that being worth it.
 *
 * Unlike {@link lcsLengthPrepared} an element the pattern does not hold cannot
 * be skipped: with `x = 0` the Myers row still advances, since `d0` picks up
 * `vn` and the horizontal deltas shift regardless.
 */
export function levenshteinPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const patternLength = prepared.length
  if (patternLength === 0) return textLength
  if (textLength === 0) return patternLength

  const words = prepared.words
  const stringText = typeof text === 'string'

  if (words === 1) {
    const masks = prepared.masks
    const highBase = prepared.highBase
    const highCount = prepared.highCount
    const highStart = prepared.highStart
    const wideOffsets = prepared.wideOffsets
    const top = 1 << (patternLength - 1)

    let vp = -1
    let vn = 0
    let distance = patternLength

    for (let i = 0; i < textLength; i++) {
      const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
      // One word, so the base is the mask's own index.
      // Written out rather than called, and that is load bearing — see the note
      // on `patternBase`, whose body this is. A single shared copy sees numbers
      // from string inputs and strings and objects from array inputs, goes
      // megamorphic, and measured 2.43x slower once other kernels had used it.
      let base = -1
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        base = symbol * 1
      } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 1
            : (wideOffsets.get(symbol) ?? -1)
      } else if (symbol === symbol) {
        base = wideOffsets.get(symbol) ?? -1
      }
      const x = base < 0 ? 0 : masks[base]

      // The addition may exceed 32 bits; `^` coerces back to int32, which
      // discards the carry-out exactly as the algorithm requires.
      const d0 = (((x & vp) + vp) ^ vp) | x | vn | 0
      let hp = vn | ~(d0 | vp)
      let hn = d0 & vp

      if ((hp & top) !== 0) distance++
      if ((hn & top) !== 0) distance--

      hp = (hp << 1) | 1
      hn = hn << 1
      vp = hn | ~(d0 | hp)
      vn = hp & d0
    }

    return distance
  }

  // The mask table and the window bounds are read once per element, so they are
  // hoisted out of the loop rather than reached through `prepared` inside it.
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  // Split by width for the reason {@link levenshteinManyWords} gives, and it
  // pays more here: 1.7x at two words, 1.8x at three and four, against text of
  // 64 to 1024 elements — this kernel does nothing but the recurrence, so the
  // row's trip through memory is a larger share of it.
  if (words === 2) {
    return preparedTwoWords(
      patternLength,
      masks,
      highBase,
      highCount,
      highStart,
      wideOffsets,
      text,
      textStart,
      textLength,
    )
  }
  if (words === 3) {
    return preparedThreeWords(
      patternLength,
      masks,
      highBase,
      highCount,
      highStart,
      wideOffsets,
      text,
      textStart,
      textLength,
    )
  }
  if (words === 4) {
    return preparedFourWords(
      patternLength,
      masks,
      highBase,
      highCount,
      highStart,
      wideOffsets,
      text,
      textStart,
      textLength,
    )
  }
  return preparedWideWords(
    patternLength,
    words,
    masks,
    highBase,
    highCount,
    highStart,
    wideOffsets,
    text,
    textStart,
    textLength,
  )
}

/**
 * Two words of held pattern, the row in locals.
 *
 * The recurrence is {@link levenshteinPrepared}'s with `w` resolved; which of
 * the two tables holds the masks is still settled once per element rather than
 * once per word, as in `lcsLengthPrepared`.
 */
function preparedTwoWords(
  patternLength: number,
  masks: Int32Array,
  highBase: number,
  highCount: number,
  highStart: number,
  wideOffsets: ReadonlyMap<unknown, number>,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  let vp0 = -1
  let vp1 = -1
  let vn0 = 0
  let vn1 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * 2
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * 2
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    const x0 = base < 0 ? 0 : masks[base]
    const x1 = base < 0 ? 0 : masks[base + 1]

    // Modular addition with the carry recovered by bit arithmetic, so no value
    // in this loop leaves the small-integer range.
    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    const addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    const carryP = hp >>> 31
    const carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0
  }

  return distance
}

/** Three words of held pattern — see {@link preparedTwoWords}. */
function preparedThreeWords(
  patternLength: number,
  masks: Int32Array,
  highBase: number,
  highCount: number,
  highStart: number,
  wideOffsets: ReadonlyMap<unknown, number>,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  let vp0 = -1
  let vp1 = -1
  let vp2 = -1
  let vn0 = 0
  let vn1 = 0
  let vn2 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * 3
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * 3
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    const x0 = base < 0 ? 0 : masks[base]
    const x1 = base < 0 ? 0 : masks[base + 1]
    const x2 = base < 0 ? 0 : masks[base + 2]

    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    let addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    let carryP = hp >>> 31
    let carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    addCarry = ((addend & vp1) | ((addend | vp1) & ~sum)) >>> 31
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0

    addend = x2 & vp2
    sum = (addend + vp2 + addCarry) | 0
    d0 = (sum ^ vp2) | x2 | vn2 | 0
    hp = vn2 | ~(d0 | vp2)
    hn = d0 & vp2
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp2 = shiftedN | ~(d0 | shiftedP)
    vn2 = shiftedP & d0
  }

  return distance
}

/** Four words of held pattern — see {@link preparedTwoWords}. */
function preparedFourWords(
  patternLength: number,
  masks: Int32Array,
  highBase: number,
  highCount: number,
  highStart: number,
  wideOffsets: ReadonlyMap<unknown, number>,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  let vp0 = -1
  let vp1 = -1
  let vp2 = -1
  let vp3 = -1
  let vn0 = 0
  let vn1 = 0
  let vn2 = 0
  let vn3 = 0
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * 4
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * 4
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    const x0 = base < 0 ? 0 : masks[base]
    const x1 = base < 0 ? 0 : masks[base + 1]
    const x2 = base < 0 ? 0 : masks[base + 2]
    const x3 = base < 0 ? 0 : masks[base + 3]

    let addend = x0 & vp0
    let sum = (addend + vp0) | 0
    let addCarry = ((addend & vp0) | ((addend | vp0) & ~sum)) >>> 31
    let d0 = (sum ^ vp0) | x0 | vn0 | 0
    let hp = vn0 | ~(d0 | vp0)
    let hn = d0 & vp0
    let shiftedP = (hp << 1) | 1
    let shiftedN = hn << 1
    let carryP = hp >>> 31
    let carryN = hn >>> 31
    vp0 = shiftedN | ~(d0 | shiftedP)
    vn0 = shiftedP & d0

    addend = x1 & vp1
    sum = (addend + vp1 + addCarry) | 0
    addCarry = ((addend & vp1) | ((addend | vp1) & ~sum)) >>> 31
    d0 = (sum ^ vp1) | x1 | vn1 | 0
    hp = vn1 | ~(d0 | vp1)
    hn = d0 & vp1
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp1 = shiftedN | ~(d0 | shiftedP)
    vn1 = shiftedP & d0

    addend = x2 & vp2
    sum = (addend + vp2 + addCarry) | 0
    addCarry = ((addend & vp2) | ((addend | vp2) & ~sum)) >>> 31
    d0 = (sum ^ vp2) | x2 | vn2 | 0
    hp = vn2 | ~(d0 | vp2)
    hn = d0 & vp2
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    carryP = hp >>> 31
    carryN = hn >>> 31
    vp2 = shiftedN | ~(d0 | shiftedP)
    vn2 = shiftedP & d0

    addend = x3 & vp3
    sum = (addend + vp3 + addCarry) | 0
    d0 = (sum ^ vp3) | x3 | vn3 | 0
    hp = vn3 | ~(d0 | vp3)
    hn = d0 & vp3
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    shiftedP = (hp << 1) | carryP
    shiftedN = (hn << 1) | carryN
    vp3 = shiftedN | ~(d0 | shiftedP)
    vn3 = shiftedP & d0
  }

  return distance
}

/**
 * Five words of held pattern or more, the row in the shared vectors.
 *
 * The last word is peeled out and an absent element takes a loop of its own,
 * for the reasons {@link levenshteinWideWords} gives.
 */
function preparedWideWords(
  patternLength: number,
  words: number,
  masks: Int32Array,
  highBase: number,
  highCount: number,
  highStart: number,
  wideOffsets: ReadonlyMap<unknown, number>,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const vp = rowVector(words)
  const vn = rowVectorN(words)
  clearRange(vp, -1, 0, words)
  clearRange(vn, 0, 0, words)

  const lastWord = words - 1
  const top = 1 << ((patternLength - 1) & WORD_MASK)
  let distance = patternLength
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    let addCarry = 0
    let carryP = 1
    let carryN = 0

    if (base < 0) {
      for (let w = 0; w < lastWord; w++) {
        const vpWord = vp[w]
        const vnWord = vn[w]

        const sum = (vpWord + addCarry) | 0
        addCarry = (vpWord & ~sum) >>> 31

        const d0 = (sum ^ vpWord) | vnWord | 0
        const hp = vnWord | ~(d0 | vpWord)
        const hn = d0 & vpWord

        const shiftedP = (hp << 1) | carryP
        const shiftedN = (hn << 1) | carryN
        carryP = hp >>> 31
        carryN = hn >>> 31

        vp[w] = shiftedN | ~(d0 | shiftedP)
        vn[w] = shiftedP & d0
      }

      const vpWord = vp[lastWord]
      const vnWord = vn[lastWord]

      const sum = (vpWord + addCarry) | 0

      const d0 = (sum ^ vpWord) | vnWord | 0
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      // See the sibling kernel: with no match bits `hn` is `vn & vp`, which is
      // zero, so there is no negative delta here to take back off.
      if ((hp & top) !== 0) distance++

      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN

      vp[lastWord] = shiftedN | ~(d0 | shiftedP)
      vn[lastWord] = shiftedP & d0
      continue
    }

    for (let w = 0; w < lastWord; w++) {
      const vpWord = vp[w]
      const vnWord = vn[w]
      const x = masks[base + w]

      const addend = x & vpWord
      const sum = (addend + vpWord + addCarry) | 0
      addCarry = ((addend & vpWord) | ((addend | vpWord) & ~sum)) >>> 31

      const d0 = (sum ^ vpWord) | x | vnWord | 0
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN
      carryP = hp >>> 31
      carryN = hn >>> 31

      vp[w] = shiftedN | ~(d0 | shiftedP)
      vn[w] = shiftedP & d0
    }

    const vpWord = vp[lastWord]
    const vnWord = vn[lastWord]
    const x = masks[base + lastWord]

    const addend = x & vpWord
    const sum = (addend + vpWord + addCarry) | 0

    const d0 = (sum ^ vpWord) | x | vnWord | 0
    const hp = vnWord | ~(d0 | vpWord)
    const hn = d0 & vpWord

    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--

    const shiftedP = (hp << 1) | carryP
    const shiftedN = (hn << 1) | carryN

    vp[lastWord] = shiftedN | ~(d0 | shiftedP)
    vn[lastWord] = shiftedP & d0
  }

  return distance
}

function shiftedPatternMatches(
  prepared: PatternMask,
  symbol: unknown,
  position: number,
): number {
  const words = prepared.words
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  // Written out rather than called — see the note on `patternBase`.
  let base = -1
  if (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  ) {
    base = symbol * words
  } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
    const shifted = symbol - highBase
    base =
      shifted >= 0 && shifted < highCount
        ? highStart + shifted * words
        : (wideOffsets.get(symbol) ?? -1)
  } else if (symbol === symbol) {
    base = wideOffsets.get(symbol) ?? -1
  }

  if (position < 0) return (base < 0 ? 0 : masks[base]) << -position

  const word = position >>> WORD_SHIFT
  if (word >= words) return 0
  const bit = position & WORD_MASK
  let matches = (base < 0 ? 0 : masks[base + word]) >>> bit

  if (bit !== 0 && word + 1 < words) {
    matches |= (base < 0 ? 0 : masks[base + word + 1]) << (WORD_BITS - bit)
  }

  return matches
}

/**
 * {@link levenshteinSmallBand} over a pattern that fits one word.
 *
 * {@link shiftedPatternMatches} windows a pattern of any width, and most of
 * what it does is decide which width it has: a word to index and a blend with
 * the word above. At one word neither applies — the mask is a single value and
 * the window is one shift of it — and
 * spelling that out measured 1.76x on Latin-1 and 1.10x on Cyrillic, with the
 * multi-word path unmoved because it never enters here. Branching on the width
 * *inside* the loop instead was tried and is much worse: it took the two- and
 * four-word cases to 0.59x.
 *
 * Only the prepared scorers reach this. {@link levenshteinUniform} tests the
 * one-word matrix before the band, so a pattern that fits a word is already
 * scored exactly by then; `preparedBandWorthwhile` has no such test, because a
 * held pattern serves the band without rebuilding anything.
 */
function smallBandOneWord(
  prepared: PatternMask,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  maximum: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  let vp = (-1 << (WORD_BITS - maximum - 1)) | 0
  let vn = 0
  let distance = maximum
  const diagonalMask = 1 << (WORD_BITS - 1)
  let horizontalMask = 1 << (WORD_BITS - 2)
  let startPosition = maximum + 1 - WORD_BITS
  const breakScore = 2 * maximum + textLength - patternLength
  const stringText = typeof text === 'string'

  let i = 0
  const diagonalEnd = Math.max(0, patternLength - maximum)
  for (; i < diagonalEnd; i++, startPosition++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * 1
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * 1
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }
    const mask = base < 0 ? 0 : masks[base]
    // The window the helper computes, with the width resolved: there is no word
    // above to blend in. No "past the word" case either, unlike the loop below:
    // `startPosition` opens at `maximum + 1 - WORD_BITS` and rises by one a row
    // for the `patternLength - maximum` rows of the diagonal, ending at
    // `patternLength - WORD_BITS` — at most zero for a pattern inside one word,
    // which is the only kind this kernel takes.
    const matches = startPosition < 0 ? mask << -startPosition : mask >>> startPosition
    const d0 = (((matches & vp) + vp) ^ vp) | matches | vn
    const hp = vn | ~(d0 | vp)
    const hn = d0 & vp
    if ((d0 & diagonalMask) === 0) distance++
    if (distance > breakScore) return maximum + 1
    vp = hn | ~((d0 >>> 1) | hp)
    vn = (d0 >>> 1) & hp
  }

  for (; i < textLength; i++, startPosition++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * 1
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * 1
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }
    const mask = base < 0 ? 0 : masks[base]
    const matches =
      startPosition < 0
        ? mask << -startPosition
        : startPosition < WORD_BITS
          ? mask >>> startPosition
          : 0
    const d0 = (((matches & vp) + vp) ^ vp) | matches | vn
    const hp = vn | ~(d0 | vp)
    const hn = d0 & vp
    if ((hp & horizontalMask) !== 0) distance++
    if ((hn & horizontalMask) !== 0) distance--
    horizontalMask >>>= 1
    if (distance > breakScore) return maximum + 1
    vp = hn | ~((d0 >>> 1) | hp)
    vn = (d0 >>> 1) & hp
  }

  return distance <= maximum ? distance : maximum + 1
}

/**
 * Upstream's Hyyrö diagonal band, adapted to JavaScript's 32-bit words.
 *
 * The band this carries is `2 * maximum + 1` wide and lives in a single word,
 * so every caller owes it `2 * maximum + 1 <= 32`, which is upstream's
 * `full_band <= 64` halved — a budget of 15 or less. A wider band is not
 * rejected here, it is truncated: the bits that fall off the word are the ones
 * furthest from the diagonal, and the answer comes back too large by however
 * many edits lived out there. Upstream states the same three bounds as
 * `assert`s, and this port keeps them as a precondition rather than a check so
 * the per-element loop stays free of it.
 *
 * The other two: `maximum` must not exceed either length, and `textLength` must
 * be at least `patternLength - maximum`.
 */
export function levenshteinSmallBand(
  prepared: PatternMask,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  maximum: number,
): number {
  if (prepared.words === 1) {
    return smallBandOneWord(prepared, patternLength, text, textStart, textLength, maximum)
  }
  let vp = (-1 << (WORD_BITS - maximum - 1)) | 0
  let vn = 0
  let distance = maximum
  const diagonalMask = 1 << (WORD_BITS - 1)
  let horizontalMask = 1 << (WORD_BITS - 2)
  let startPosition = maximum + 1 - WORD_BITS
  const breakScore = 2 * maximum + textLength - patternLength
  const stringText = typeof text === 'string'

  let i = 0
  const diagonalEnd = Math.max(0, patternLength - maximum)
  for (; i < diagonalEnd; i++, startPosition++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const matches = shiftedPatternMatches(prepared, symbol, startPosition)
    const d0 = (((matches & vp) + vp) ^ vp) | matches | vn
    const hp = vn | ~(d0 | vp)
    const hn = d0 & vp
    if ((d0 & diagonalMask) === 0) distance++
    if (distance > breakScore) return maximum + 1
    vp = hn | ~((d0 >>> 1) | hp)
    vn = (d0 >>> 1) & hp
  }

  for (; i < textLength; i++, startPosition++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    const matches = shiftedPatternMatches(prepared, symbol, startPosition)
    const d0 = (((matches & vp) + vp) ^ vp) | matches | vn
    const hp = vn | ~(d0 | vp)
    const hn = d0 & vp
    if ((hp & horizontalMask) !== 0) distance++
    if ((hn & horizontalMask) !== 0) distance--
    horizontalMask >>>= 1
    if (distance > breakScore) return maximum + 1
    vp = hn | ~((d0 >>> 1) | hp)
    vn = (d0 >>> 1) & hp
  }

  return distance <= maximum ? distance : maximum + 1
}

/**
 * Uniform Levenshtein distance.
 *
 * Unlike the LCS the recurrence is not symmetric in cost, but the distance is,
 * so either side can serve as the pattern.
 */
export function levenshteinUniform(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number = Number.MAX_SAFE_INTEGER,
  scoreHint: number = scoreCutoff,
): number {
  if (s1.length === 0) return s2.length
  if (s2.length === 0) return s1.length

  // Ahead of the affix scan, because trimming cannot change this answer: a
  // common prefix and suffix are removed from both sides in equal counts, so
  // the trimmed difference is the untrimmed one. What the two `.length`s say
  // about reachability, they say without reading an element — and the pair this
  // rejects is the one the scan is worst on, since a choice that shares a long
  // prefix with the query and misses the cutoff on length alone had that whole
  // prefix walked before anything looked at the lengths. Measured at 64x over
  // 1024 elements against a 256-element prefix of themselves, and 5x on the
  // same lengths sharing nothing at all.
  const lengthDifference = Math.abs(s1.length - s2.length)
  if (lengthDifference > scoreCutoff) return scoreCutoff + 1

  // Allocating the shared table is deferred to the three kernels that read it,
  // rather than paid by every comparison. `measureAffix` compares the two
  // sequences directly, and the outcomes reachable before those kernels —
  // an identical pair, a length rejection, `levenshteinMbleven`, and
  // `levenshteinSmallBand` over a pattern of its own — read no shared state at
  // all. It cost about 2.4ns, which is 5% of an eight-element comparison, and
  // `extract` over short choices is made of those.
  //
  // Load bearing where it survives: `buildWordMasks` and `buildBlockMasks`
  // return without building if the table is absent, so a kernel that ran ahead
  // of this would score against no masks and answer `patternLength`.
  measureAffix(s1, 0, s1.length, s2, 0, s2.length)
  const prefix = affixPrefix
  const len1 = affixLen1
  const len2 = affixLen2

  if (len1 === 0) return len2
  if (len2 === 0) return len1

  // Left above the one-word branch, which never calls it. Moving it below was
  // measured: the difference does not clear the noise, in the suite or in
  // isolation, so the dispatcher keeps the order that reads in one direction.
  let bandPattern: PatternMask | null = null
  // Every budget below is at least `lengthDifference`: the dispatcher rejected
  // a larger difference above, and `hinted` opens at the difference itself.
  const bounded = (budget: number): number => {
    if (budget < 4 && (budget | 0) === budget) {
      return levenshteinMbleven(s1, prefix, len1, s2, prefix, len2, budget)
    }
    // The diagonal band is `2 * budget + 1` wide and the kernel holds it in a
    // single word — upstream's `full_band <= 64`, halved for our 32-bit words,
    // which bounds the budget at 15.
    //
    // Upstream spells that test `min(len1, 2 * budget + 1) <= 64`, but only
    // reaches it once the whole matrix has failed to fit a word, so the `min`
    // never picks `len1` there. Ported without that ordering it did, and a band
    // wider than a word went to a kernel that silently drops what does not fit:
    // a 32-element input 23 edits from a 33-element one answered 24 at a budget
    // of 30. The one-word matrix below now takes those, as upstream does.
    if (
      2 * budget + 1 <= WORD_BITS &&
      budget <= len1 &&
      budget <= len2 &&
      len2 >= len1 - budget
    ) {
      bandPattern ??= preparePattern(s1, prefix, len1)
      return levenshteinSmallBand(bandPattern, len1, s2, prefix, len2, budget)
    }
    // The band is set by the budget rather than by the lengths, so the work is
    // rows times band width: the shorter input is the one worth streaming.
    return len1 >= len2
      ? levenshteinManyWordsBanded(s1, prefix, len1, s2, prefix, len2, budget)
      : levenshteinManyWordsBanded(s2, prefix, len2, s1, prefix, len1, budget)
  }

  // A cutoff no smaller than the longest input cannot reject anything, so the
  // unbounded kernel below is both correct and faster than banding to a width
  // that spans the whole matrix. Every similarity-shaped scorer converts its
  // cutoff into a distance budget of exactly that size when the caller asks for
  // no cutoff at all, which is the common case rather than a corner one.
  const longest = Math.max(len1, len2)
  const cutoff = Math.min(Math.floor(scoreCutoff), longest)

  // A pattern of a word or less puts the whole matrix in one word, so there is
  // no band to widen towards: a single pass over the longer side settles the
  // exact distance, whatever the budget. Upstream tests this ahead of the band
  // for that reason, and the order is load bearing — the band test above reads
  // as if a short pattern narrowed the band, and it does not.
  //
  // Budgets under four keep going: `levenshteinMbleven` answers those by
  // comparing elements directly, without building a mask at all.
  if (Math.min(len1, len2) <= WORD_BITS && cutoff >= 4) {
    const distance =
      len1 <= len2
        ? levenshteinOneWord(s1, prefix, len1, s2, prefix, len2)
        : levenshteinOneWord(s2, prefix, len2, s1, prefix, len1)
    return distance <= cutoff ? distance : cutoff + 1
  }

  // Try the caller's estimate first, widening geometrically until the cutoff.
  // A failed narrow run only says the distance is larger than that run's band;
  // the final run (or unbounded bit-vector kernel) still determines the result.
  //
  // The floor at a full word is a bound on the worst case, not a claim that the
  // narrower widths run the same kernel — they do not. Under four is
  // `levenshteinMbleven`, and four to fifteen is `levenshteinSmallBand`, whose
  // `2 * budget + 1` band still fits one word.
  //
  // Honouring a hint below it was measured and is worse. A hint is an estimate,
  // and an optimistic one buys a whole extra pass: at `scoreCutoff: 16,
  // scoreHint: 8` over 4096 elements, the small-band pass scans the input,
  // fails, and the banded kernel then scans it again — 0.64x. The win needs the
  // hint to be accurate; the loss only needs it to be low. Only `mbleven`
  // budgets under four are cheap enough to gamble on, and they measured level,
  // so there is nothing to collect there either.
  let hinted = Math.max(lengthDifference, Math.floor(scoreHint), WORD_BITS - 1)
  if (Number.isFinite(hinted) && hinted < cutoff) {
    while (hinted < cutoff) {
      const result = bounded(hinted)
      if (result <= hinted) return result
      hinted = Math.min(cutoff, hinted * 2 + 1)
    }
  }
  if (cutoff < longest) return bounded(cutoff)

  const words1 = (len1 + WORD_MASK) >>> WORD_SHIFT
  const words2 = (len2 + WORD_MASK) >>> WORD_SHIFT
  const firstIsPattern = words1 * len2 <= words2 * len1

  const pattern = firstIsPattern ? s1 : s2
  const text = firstIsPattern ? s2 : s1
  const patternLength = firstIsPattern ? len1 : len2
  const textLength = firstIsPattern ? len2 : len1

  return patternLength <= WORD_BITS
    ? levenshteinOneWord(pattern, prefix, patternLength, text, prefix, textLength)
    : levenshteinManyWords(pattern, prefix, patternLength, text, prefix, textLength)
}
