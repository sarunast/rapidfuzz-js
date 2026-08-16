import {
  affixLen1,
  affixLen2,
  affixPrefix,
  directSlots,
  directStamps,
  blockMasksFor,
  buildWordMasks,
  clearRange,
  directLimit,
  measureAffix,
  rowVector,
  wordCount,
  type BuiltMasks,
} from '../../bitmask/blockMasks.js'
import type { PatternMask } from '../../bitmask/pattern.js'

const WORD_BITS = 32
const WORD_SHIFT = 5
const DIRECT_LOOKUP_LIMIT = 256

function popcount(word: number): number {
  let bits = word - ((word >>> 1) & 0x5555_5555)
  bits = (bits & 0x3333_3333) + ((bits >>> 2) & 0x3333_3333)
  return (((bits + (bits >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24
}

function lcsOneWord(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  return lcsOneWordStamped(
    buildWordMasks(pattern, patternStart, patternLength),
    text,
    textStart,
    textLength,
  )
}

function lcsOneWordStamped(
  masks: BuiltMasks,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const stamp = masks.stamp
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

  let s = -1
  const limit = directLimit

  if (typeof text === 'string') {
    for (let i = 0; i < textLength; i++) {
      const symbol = text.charCodeAt(textStart + i)
      const matches = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : 0

      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
    }

    return popcount(~s)
  }

  for (let i = 0; i < textLength; i++) {
    const symbol = text[textStart + i]
    let matches: number

    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      matches = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < limit &&
      (symbol | 0) === symbol
    ) {
      matches = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (symbol === symbol) {
      matches = wide.get(symbol) ?? 0
    } else {
      matches = 0
    }

    const u = s & matches
    s = (s + u) | 0 | (s & ~u)
  }

  return popcount(~s)
}

function lcsManyWords(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const words = wordCount(patternLength)
  const masks = blockMasksFor(pattern, patternStart, patternLength, words)

  if (words === 4) return lcsFourWordsStamped(masks, text, textStart, textLength)
  return lcsManyWordsStamped(masks, words, text, textStart, textLength)
}

function lcsFourWordsStamped(
  masks: BuiltMasks,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

  let s0 = -1
  let s1 = -1
  let s2 = -1
  let s3 = -1
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    let offset: number

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else {
        offset = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : -1
      }
    } else {
      const symbol = text[textStart + i]
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (
        typeof symbol === 'number' &&
        symbol >= DIRECT_LOOKUP_LIMIT &&
        symbol < limit &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (symbol === symbol) {
        offset = wide.get(symbol) ?? -1
      } else {
        offset = -1
      }
    }

    if (offset < 0) continue

    // Modular addition with the carry recovered by bit arithmetic, so no value
    // in this loop leaves the small-integer range. Only the addition needs a
    // carry chain — no borrow can leave a word. Repeated at every width below.
    let u = s0 & pool[offset]
    let sum = (s0 + u) | 0
    let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & pool[offset + 1]
    sum = (s1 + u + carry) | 0
    carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
    s1 = sum | (s1 & ~u)

    u = s2 & pool[offset + 2]
    sum = (s2 + u + carry) | 0
    carry = ((s2 & u) | ((s2 | u) & ~sum)) >>> 31
    s2 = sum | (s2 & ~u)

    u = s3 & pool[offset + 3]
    sum = (s3 + u + carry) | 0
    s3 = sum | (s3 & ~u)
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2) + popcount(~s3)
}

function lcsManyWordsStamped(
  masks: BuiltMasks,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const row = rowVector(words)
  clearRange(row, -1, 0, words)

  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    let offset: number

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else {
        offset = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : -1
      }
    } else {
      const symbol = text[textStart + i]
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (
        typeof symbol === 'number' &&
        symbol >= DIRECT_LOOKUP_LIMIT &&
        symbol < limit &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (symbol === symbol) {
        offset = wide.get(symbol) ?? -1
      } else {
        offset = -1
      }
    }

    if (offset < 0) continue

    let carry = 0
    let w = 0

    const unrolledEnd = words & ~7
    for (; w < unrolledEnd; w += 8) {
      let s = row[w]
      let u = s & pool[offset + w]
      let sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w] = sum | (s & ~u)

      s = row[w + 1]
      u = s & pool[offset + w + 1]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 1] = sum | (s & ~u)

      s = row[w + 2]
      u = s & pool[offset + w + 2]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 2] = sum | (s & ~u)

      s = row[w + 3]
      u = s & pool[offset + w + 3]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 3] = sum | (s & ~u)

      s = row[w + 4]
      u = s & pool[offset + w + 4]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 4] = sum | (s & ~u)

      s = row[w + 5]
      u = s & pool[offset + w + 5]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 5] = sum | (s & ~u)

      s = row[w + 6]
      u = s & pool[offset + w + 6]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 6] = sum | (s & ~u)

      s = row[w + 7]
      u = s & pool[offset + w + 7]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 7] = sum | (s & ~u)
    }

    for (; w < words; w++) {
      const s = row[w]
      const u = s & pool[offset + w]

      const sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31

      row[w] = sum | (s & ~u)
    }
  }

  let count = 0
  for (let w = 0; w < words; w++) count += popcount(~row[w])
  return count
}

function lcsManyWordsBanded(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const words = wordCount(patternLength)
  const masks = blockMasksFor(pattern, patternStart, patternLength, words)
  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stamp = masks.stamp
  const pool = masks.pool
  const wide = masks.wide
  const slots = directSlots()
  const stamps = directStamps()

  const left = patternLength - required
  const right = textLength - required
  let firstWord = 0
  let lastWord = Math.min(words, Math.ceil((left + 1) / WORD_BITS))
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    let offset = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      if (stamps[symbol] === stamp) offset = slots[symbol]
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < limit &&
      (symbol | 0) === symbol
    ) {
      if (stamps[symbol] === stamp) offset = slots[symbol]
    } else if (symbol === symbol) {
      offset = wide.get(symbol) ?? -1
    }

    if (offset >= 0) {
      let carry = 0

      for (let word = firstWord; word < lastWord; word++) {
        const s = row[word]
        const u = s & pool[offset + word]
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[word] = sum | (s & ~u)
      }
    }

    if ((i & 63) === 63) {
      let possible = textLength - i - 1
      for (let word = 0; word < lastWord; word++) possible += popcount(~row[word])
      if (possible < required) return 0
    }

    if (i > right) firstWord = Math.floor((i - right) / WORD_BITS)
    if (i + 1 + left <= patternLength) {
      lastWord = Math.min(words, ((i + 1 + left) >>> WORD_SHIFT) + 1)
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  return count >= required ? count : 0
}

const MBLEVEN_OPS: ReadonlyArray<readonly number[]> = [
  [], // d = 0 — parity rules this out
  [0x01], // d = 1
  [0x09, 0x06], // d = 0
  [0x01], // d = 1
  [0x05], // d = 2
  [0x09, 0x06], // d = 0
  [0x25, 0x19, 0x16], // d = 1
  [0x05], // d = 2
  [0x15], // d = 3
  [0x96, 0x66, 0x5a, 0x99, 0x69, 0xa5], // d = 0
  [0x25, 0x19, 0x16], // d = 1
  [0x65, 0x56, 0x95, 0x59], // d = 2
  [0x15], // d = 3
  [0x55], // d = 4
]

const MBLEVEN_LIMIT = 4

function lcsMbleven(
  longer: ArrayLike<unknown>,
  longerStart: number,
  longerLength: number,
  shorter: ArrayLike<unknown>,
  shorterStart: number,
  shorterLength: number,
  budget: number,
): number {
  const lengthDiff = longerLength - shorterLength
  const scripts = MBLEVEN_OPS[(budget + budget * budget) / 2 + lengthDiff - 1]
  let best = 0

  for (let s = 0; s < scripts.length; s++) {
    let script = scripts[s]
    let i = 0
    let j = 0
    let common = 0

    while (i < longerLength && j < shorterLength) {
      if (longer[longerStart + i] !== shorter[shorterStart + j]) {
        if (script === 0) break

        if ((script & 1) !== 0) i++
        else j++
        script >>>= 2
      } else {
        common++
        i++
        j++
      }
    }

    if (common > best) best = common
  }

  return best
}

export function lcsLengthRange(
  s1: ArrayLike<unknown>,
  start1: number,
  len1: number,
  s2: ArrayLike<unknown>,
  start2: number,
  len2: number,
  budget: number,
): number {
  if (len1 === 0 || len2 === 0) return 0

  if (budget < (len1 < len2 ? len2 - len1 : len1 - len2)) return 0

  measureAffix(s1, start1, len1, s2, start2, len2)
  const prefix = affixPrefix
  const middle1 = affixLen1
  const middle2 = affixLen2

  const common = len1 - middle1
  if (middle1 === 0 || middle2 === 0) return common

  const lengthDiff = middle1 < middle2 ? middle2 - middle1 : middle1 - middle2

  if (budget < lengthDiff || budget < 1) return common

  if (budget <= MBLEVEN_LIMIT && (budget | 0) === budget) {
    const firstIsLonger = middle1 >= middle2

    return (
      common +
      lcsMbleven(
        firstIsLonger ? s1 : s2,
        (firstIsLonger ? start1 : start2) + prefix,
        firstIsLonger ? middle1 : middle2,
        firstIsLonger ? s2 : s1,
        (firstIsLonger ? start2 : start1) + prefix,
        firstIsLonger ? middle2 : middle1,
        budget,
      )
    )
  }

  const firstIsPattern =
    middle1 <= WORD_BITS && middle2 <= WORD_BITS
      ? middle2 <= middle1
      : wordCount(middle1) * middle2 <= wordCount(middle2) * middle1

  const pattern = firstIsPattern ? s1 : s2
  const patternStart = (firstIsPattern ? start1 : start2) + prefix
  const patternLength = firstIsPattern ? middle1 : middle2
  const text = firstIsPattern ? s2 : s1
  const textStart = (firstIsPattern ? start2 : start1) + prefix
  const textLength = firstIsPattern ? middle2 : middle1

  const requiredTotal = Math.max(0, Math.ceil((len1 + len2 - budget) / 2))
  const requiredMiddle = Math.max(0, requiredTotal - common)
  const fullBand = patternLength + textLength - 2 * requiredMiddle + 1
  const bandWords = Math.min(
    wordCount(patternLength),
    Math.floor(fullBand / WORD_BITS) + 2,
  )

  const middle =
    patternLength <= WORD_BITS
      ? lcsOneWord(pattern, patternStart, patternLength, text, textStart, textLength)
      : requiredMiddle > 0 && bandWords < wordCount(patternLength)
        ? lcsManyWordsBanded(
            pattern,
            patternStart,
            patternLength,
            text,
            textStart,
            textLength,
            requiredMiddle,
          )
        : lcsManyWords(pattern, patternStart, patternLength, text, textStart, textLength)

  return common + middle
}

export function lcsLengthPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const words = prepared.words
  if (words === 0 || textLength === 0) return 0
  if (words === 4) return lcsFourWordsPrepared(prepared, text, textStart, textLength)
  if (words === 2) return lcsTwoWordsPrepared(prepared, text, textStart, textLength)
  if (words === 3) return lcsThreeWordsPrepared(prepared, text, textStart, textLength)

  if (words === 1) {
    let s = -1
    const masks = prepared.masks
    const highBase = prepared.highBase
    const highCount = prepared.highCount
    const highStart = prepared.highStart
    const wideOffsets = prepared.wideOffsets

    if (typeof text === 'string') {
      for (let i = 0; i < textLength; i++) {
        const symbol = text.charCodeAt(textStart + i)
        let base: number
        if (symbol < DIRECT_LOOKUP_LIMIT) {
          base = symbol
        } else {
          const shifted = symbol - highBase
          base =
            shifted >= 0 && shifted < highCount
              ? highStart + shifted
              : (wideOffsets.get(symbol) ?? -1)
        }
        const matches = base < 0 ? 0 : masks[base]
        const u = s & matches
        s = (s + u) | 0 | (s & ~u)
      }
      return popcount(~s)
    }

    for (let i = 0; i < textLength; i++) {
      const symbol = text[textStart + i]
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
      const matches = base < 0 ? 0 : masks[base]
      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
    }
    return popcount(~s)
  }

  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
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
    if (base < 0) continue

    let s = row[0]
    let u = s & masks[base]
    let sum = (s + u) | 0
    let carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[0] = sum | (s & ~u)

    s = row[1]
    u = s & masks[base + 1]
    sum = (s + u + carry) | 0
    carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[1] = sum | (s & ~u)

    s = row[2]
    u = s & masks[base + 2]
    sum = (s + u + carry) | 0
    carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[2] = sum | (s & ~u)

    for (let w = 3; w < words; w++) {
      s = row[w]
      u = s & masks[base + w]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w] = sum | (s & ~u)
    }
  }

  let count = 0
  for (let w = 0; w < words; w++) count += popcount(~row[w])
  return count
}

function lcsTwoWordsPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 2
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 2
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
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
    }
    if (base < 0) continue

    let u = s0 & masks[base]
    let sum = (s0 + u) | 0
    const carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & masks[base + 1]
    sum = (s1 + u + carry) | 0
    s1 = sum | (s1 & ~u)
  }

  return popcount(~s0) + popcount(~s1)
}

function lcsThreeWordsPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1
  let s2 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 3
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 3
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
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
    }
    if (base < 0) continue

    let u = s0 & masks[base]
    let sum = (s0 + u) | 0
    let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & masks[base + 1]
    sum = (s1 + u + carry) | 0
    carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
    s1 = sum | (s1 & ~u)

    u = s2 & masks[base + 2]
    sum = (s2 + u + carry) | 0
    s2 = sum | (s2 & ~u)
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2)
}

function lcsFourWordsPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1
  let s2 = -1
  let s3 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 4
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 4
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
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
    }
    if (base < 0) continue

    let u = s0 & masks[base]
    let sum = (s0 + u) | 0
    let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & masks[base + 1]
    sum = (s1 + u + carry) | 0
    carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
    s1 = sum | (s1 & ~u)

    u = s2 & masks[base + 2]
    sum = (s2 + u + carry) | 0
    carry = ((s2 & u) | ((s2 | u) & ~sum)) >>> 31
    s2 = sum | (s2 & ~u)

    u = s3 & masks[base + 3]
    sum = (s3 + u + carry) | 0
    s3 = sum | (s3 & ~u)
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2) + popcount(~s3)
}

function lcsPreparedBanded(
  prepared: PatternMask,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  const left = prepared.length - required
  const right = textLength - required
  let firstWord = 0
  let lastWord = Math.min(words, Math.ceil((left + 1) / WORD_BITS))

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

    if (base >= 0) {
      let carry = 0

      for (let word = firstWord; word < lastWord; word++) {
        const s = row[word]
        const u = s & masks[base + word]
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[word] = sum | (s & ~u)
      }
    }

    if ((i & 15) === 15) {
      let possible = textLength - i - 1
      for (let word = 0; word < lastWord; word++) possible += popcount(~row[word])
      if (possible < required) return -1
    }

    if (i > right) firstWord = (i - right) >>> WORD_SHIFT
    if (i + 1 + left <= prepared.length) {
      lastWord = Math.min(words, ((i + 1 + left) >>> WORD_SHIFT) + 1)
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  return count >= required ? count : -1
}

function lcsTwoWordsPreparedBounded(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 2
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 2
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
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
    }

    if (base >= 0) {
      let u = s0 & masks[base]
      let sum = (s0 + u) | 0
      const carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
      s0 = sum | (s0 & ~u)

      u = s1 & masks[base + 1]
      sum = (s1 + u + carry) | 0
      s1 = sum | (s1 & ~u)
    }

    if ((i & 7) === 7 || i + 1 === textLength) {
      if (popcount(~s0) + popcount(~s1) + textLength - i - 1 < required) return -1
    }
  }

  return popcount(~s0) + popcount(~s1)
}

function lcsThreeWordsPreparedBounded(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1
  let s2 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 3
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 3
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
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
    }

    if (base >= 0) {
      let u = s0 & masks[base]
      let sum = (s0 + u) | 0
      let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
      s0 = sum | (s0 & ~u)

      u = s1 & masks[base + 1]
      sum = (s1 + u + carry) | 0
      carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
      s1 = sum | (s1 & ~u)

      u = s2 & masks[base + 2]
      sum = (s2 + u + carry) | 0
      s2 = sum | (s2 & ~u)
    }

    if ((i & 7) === 7 || i + 1 === textLength) {
      if (popcount(~s0) + popcount(~s1) + popcount(~s2) + textLength - i - 1 < required) {
        return -1
      }
    }
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2)
}

export function lcsLengthPreparedBounded(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const words = prepared.words
  if (words === 0 || textLength === 0) return required > 0 ? -1 : 0

  if (required > prepared.length || required > textLength) return -1

  if (words === 1) {
    let s = -1
    const masks = prepared.masks
    const highBase = prepared.highBase
    const highCount = prepared.highCount
    const highStart = prepared.highStart
    const wideOffsets = prepared.wideOffsets

    if (typeof text === 'string') {
      for (let i = 0; i < textLength; i++) {
        const symbol = text.charCodeAt(textStart + i)
        let base: number
        if (symbol < DIRECT_LOOKUP_LIMIT) {
          base = symbol
        } else {
          const shifted = symbol - highBase
          base =
            shifted >= 0 && shifted < highCount
              ? highStart + shifted
              : (wideOffsets.get(symbol) ?? -1)
        }
        const matches = base < 0 ? 0 : masks[base]
        const u = s & matches
        s = (s + u) | 0 | (s & ~u)
        if (
          ((i & 7) === 7 || i + 1 === textLength) &&
          popcount(~s) + textLength - i - 1 < required
        )
          return -1
      }
      return popcount(~s)
    }

    for (let i = 0; i < textLength; i++) {
      const symbol = text[textStart + i]
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
      const matches = base < 0 ? 0 : masks[base]
      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
      if (
        ((i & 7) === 7 || i + 1 === textLength) &&
        popcount(~s) + textLength - i - 1 < required
      )
        return -1
    }
    return popcount(~s)
  }

  const fullBand = prepared.length + textLength - 2 * required + 1
  const bandWords = Math.min(words, Math.floor(fullBand / WORD_BITS) + 2)
  if (required > 0 && bandWords < words) {
    return lcsPreparedBanded(prepared, words, text, textStart, textLength, required)
  }

  if (words === 2) {
    return lcsTwoWordsPreparedBounded(prepared, text, textStart, textLength, required)
  }
  if (words === 3) {
    return lcsThreeWordsPreparedBounded(prepared, text, textStart, textLength, required)
  }

  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
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

    if (base >= 0) {
      let s = row[0]
      let u = s & masks[base]
      let sum = (s + u) | 0
      let carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[0] = sum | (s & ~u)

      s = row[1]
      u = s & masks[base + 1]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[1] = sum | (s & ~u)

      s = row[2]
      u = s & masks[base + 2]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[2] = sum | (s & ~u)

      for (let word = 3; word < words; word++) {
        s = row[word]
        u = s & masks[base + word]
        sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[word] = sum | (s & ~u)
      }
    }

    if ((i & 7) === 7 || i + 1 === textLength) {
      let possible = textLength - i - 1
      for (let word = 0; word < words; word++) possible += popcount(~row[word])
      if (possible < required) return -1
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  return count
}
