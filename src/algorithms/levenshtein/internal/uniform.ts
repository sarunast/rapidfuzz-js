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
  measureAffix,
  rowVector,
  rowVectorN,
  type BuiltMasks,
} from '../../bitmask/blockMasks.js'
import { preparePattern, type PatternMask } from '../../bitmask/pattern.js'
import { wordCount } from '../../bitmask/words.js'

// Copies of the canonical definitions in `blockMasks.ts`; a module-local const
// folds where a cross-module binding does not. Any copy that disagrees is a bug.
const WORD_BITS = 32
const WORD_SHIFT = 5
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

/**
 * Bits above `patternLength` stay set: they describe pattern positions that do
 * not exist, and carries only travel upward, so they can never influence the
 * score read off bit `patternLength - 1`.
 */
function levenshteinOneWord(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const masks = buildWordMasks(pattern, patternStart, patternLength)
  const stamp = masks.stamp
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

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

    // The addition may exceed 32 bits; `^` coerces back to int32, discarding
    // the carry-out exactly as the algorithm requires.
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

function patternOffset(
  symbol: unknown,
  stamp: number,
  slots: Int32Array,
  stamps: Int32Array,
  wide: ReadonlyMap<unknown, number>,
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
  const words = wordCount(patternLength)
  const masks = blockMasksFor(pattern, patternStart, patternLength, words)

  if (words === 2) {
    return levenshteinTwoWords(patternLength, text, textStart, textLength, masks)
  }
  if (words === 3) {
    return levenshteinThreeWords(patternLength, text, textStart, textLength, masks)
  }
  if (words === 4) {
    return levenshteinFourWords(patternLength, text, textStart, textLength, masks)
  }
  return levenshteinWideWords(patternLength, words, text, textStart, textLength, masks)
}

function levenshteinTwoWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  masks: BuiltMasks,
): number {
  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

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
    // in this loop leaves the small-integer range. Every multi-word kernel
    // below, and in `lcs/internal/kernel.ts`, repeats this shape.
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

function levenshteinThreeWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  masks: BuiltMasks,
): number {
  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

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

function levenshteinFourWords(
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  masks: BuiltMasks,
): number {
  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

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

function levenshteinWideWords(
  patternLength: number,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  masks: BuiltMasks,
): number {
  const vp = rowVector(words)
  const vn = rowVectorN(words)
  clearRange(vp, -1, 0, words)
  clearRange(vn, 0, 0, words)

  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

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

    // The absent-element arm is the same recurrence with `x` folded to zero.
    // Worth 1.11x on 1024-element pairs (measured 2026-08-15); collapsing the
    // two into one loop with a per-word ternary loses that.
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

function levenshteinManyWordsBanded(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  budget: number,
): number {
  const words = wordCount(patternLength)
  const masks = blockMasksFor(pattern, patternStart, patternLength, words)

  const vp = rowVector(words)
  const vn = rowVectorN(words)
  const scores = bandVector(words)

  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

  const last = 1 << ((patternLength - 1) & WORD_MASK)
  const stringText = typeof text === 'string'

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

  const opening = lastWord + 1
  clearRange(vp, -1, 0, opening)
  clearRange(vn, 0, 0, opening)
  for (let i = 0; i < opening; i++) {
    scores[i] = i + 1 === words ? patternLength : (i + 1) * WORD_BITS
  }

  let carryP = 1
  let carryN = 0
  const limit = directLimit

  for (let row = 0; row < textLength; row++) {
    const symbol = stringText ? text.charCodeAt(textStart + row) : text[textStart + row]
    const offset = patternOffset(symbol, stamp, slots, stamps, wide, limit)

    carryP = 1
    carryN = 0
    for (let word = firstWord; word <= lastWord; word++) {
      const matches = offset < 0 ? 0 : pool[offset + word]
      const vpWord = vp[word]
      const vnWord = vn[word]

      const x = matches | carryN
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

        const matches = offset < 0 ? 0 : pool[offset + lastWord]
        const vpWord = vp[lastWord]
        const vnWord = vn[lastWord]

        const x = matches | carryN
        const d0 = ((((x & vpWord) + vpWord) | 0) ^ vpWord) | x | vnWord
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

    if (lastWord < firstWord) return budget + 1
  }

  return scores[words - 1]
}

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

export function levenshteinPreparedRow(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  textStep: number,
  out: Uint32Array,
): void {
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

  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

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

export function levenshteinUniform(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number = Number.MAX_SAFE_INTEGER,
  scoreHint: number = scoreCutoff,
): number {
  if (s1.length === 0) return s2.length
  if (s2.length === 0) return s1.length

  const lengthDifference = Math.abs(s1.length - s2.length)
  if (lengthDifference > scoreCutoff) return scoreCutoff + 1

  measureAffix(s1, 0, s1.length, s2, 0, s2.length)
  const prefix = affixPrefix
  const len1 = affixLen1
  const len2 = affixLen2

  if (len1 === 0) return len2
  if (len2 === 0) return len1

  let bandPattern: PatternMask | null = null
  const bounded = (budget: number): number => {
    if (budget < 4 && (budget | 0) === budget) {
      return levenshteinMbleven(s1, prefix, len1, s2, prefix, len2, budget)
    }
    if (
      2 * budget + 1 <= WORD_BITS &&
      budget <= len1 &&
      budget <= len2 &&
      len2 >= len1 - budget
    ) {
      bandPattern ??= preparePattern(s1, prefix, len1)
      return levenshteinSmallBand(bandPattern, len1, s2, prefix, len2, budget)
    }
    return len1 >= len2
      ? levenshteinManyWordsBanded(s1, prefix, len1, s2, prefix, len2, budget)
      : levenshteinManyWordsBanded(s2, prefix, len2, s1, prefix, len1, budget)
  }

  const longest = Math.max(len1, len2)
  const cutoff = Math.min(Math.floor(scoreCutoff), longest)

  if (Math.min(len1, len2) <= WORD_BITS && cutoff >= 4) {
    const distance =
      len1 <= len2
        ? levenshteinOneWord(s1, prefix, len1, s2, prefix, len2)
        : levenshteinOneWord(s2, prefix, len2, s1, prefix, len1)
    return distance <= cutoff ? distance : cutoff + 1
  }

  let hinted = Math.max(lengthDifference, Math.floor(scoreHint), WORD_BITS - 1)
  if (Number.isFinite(hinted) && hinted < cutoff) {
    while (hinted < cutoff) {
      const result = bounded(hinted)
      if (result <= hinted) return result
      hinted = Math.min(cutoff, hinted * 2 + 1)
    }
  }
  if (cutoff < longest) return bounded(cutoff)

  const words1 = wordCount(len1)
  const words2 = wordCount(len2)
  const firstIsPattern = words1 * len2 <= words2 * len1

  const pattern = firstIsPattern ? s1 : s2
  const text = firstIsPattern ? s2 : s1
  const patternLength = firstIsPattern ? len1 : len2
  const textLength = firstIsPattern ? len2 : len1

  return patternLength <= WORD_BITS
    ? levenshteinOneWord(pattern, prefix, patternLength, text, prefix, textLength)
    : levenshteinManyWords(pattern, prefix, patternLength, text, prefix, textLength)
}
