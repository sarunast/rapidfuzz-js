import { wordCount } from '../../bitmask/blockMasks.js'
import {
  oneWordMasks,
  popcount32,
  validBits,
  wordPositionMasks,
} from '../../bitmask/positionMasks.js'

interface LcsSeqMatrix {
  readonly sim: number
  readonly rows: Int32Array
  readonly words: number
}

export function lcsSeqMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
): LcsSeqMatrix {
  if (s1Length === 0 || s2Length === 0) {
    return { sim: 0, rows: new Int32Array(0), words: 0 }
  }

  const words = wordCount(s1Length)
  if (words === 1) {
    return oneWordLcsSeqMatrix(s1, s1Start, s1Length, s2, s2Start, s2Length)
  }

  const { masks, bases, minSymbol, maxSymbol, wide } = wordPositionMasks(
    s1,
    s1Start,
    s1Length,
    words,
  )
  const stringText = typeof s2 === 'string'
  if (words > 32) {
    const rows = new Int32Array(s2Length * words)
    const state = new Int32Array(words).fill(-1)

    for (let j = 0; j < s2Length; j++) {
      const index = s2Start + j
      const symbol = stringText ? s2.charCodeAt(index) : s2[index]
      const base =
        typeof symbol === 'number' &&
        symbol >= minSymbol &&
        symbol <= maxSymbol &&
        (symbol | 0) === symbol
          ? bases[symbol - minSymbol]
          : wide === null
            ? 0
            : (wide.get(symbol) ?? 0)

      if (base !== 0) {
        let carry = 0
        for (let w = 0; w < words; w++) {
          const s = state[w]
          const u = s & masks[base + w]
          const sum = (s + u + carry) | 0
          carry = ((s & u) | ((s | u) & ~sum)) >>> 31
          state[w] = sum | (s & ~u)
        }
      }

      rows.set(state, j * words)
    }

    let sim = 0
    for (let w = 0; w < words; w++) {
      const valid = w === words - 1 ? validBits(s1Length) : -1
      sim += popcount32(~state[w] & valid)
    }

    return { sim, rows, words }
  }

  const storage = new Int32Array((s2Length + 1) * words)
  storage.fill(-1, 0, words)
  const rows = storage.subarray(words)

  for (let j = 0; j < s2Length; j++) {
    const index = s2Start + j
    const symbol = stringText ? s2.charCodeAt(index) : s2[index]
    const previousBase = j * words
    const rowBase = previousBase + words
    const base =
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
        ? bases[symbol - minSymbol]
        : wide === null
          ? 0
          : (wide.get(symbol) ?? 0)

    if (base !== 0) {
      let carry = 0
      for (let w = 0; w < words; w++) {
        const s = storage[previousBase + w]
        const u = s & masks[base + w]
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        storage[rowBase + w] = sum | (s & ~u)
      }
    } else {
      storage.copyWithin(rowBase, previousBase, rowBase)
    }
  }

  let sim = 0
  for (let w = 0; w < words; w++) {
    const valid = w === words - 1 ? validBits(s1Length) : -1
    sim += popcount32(~storage[s2Length * words + w] & valid)
  }

  return { sim, rows, words }
}

function oneWordLcsSeqMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
): LcsSeqMatrix {
  const { direct, minSymbol, maxSymbol, wide } = oneWordMasks(s1, s1Start, s1Length)
  const stringText = typeof s2 === 'string'
  const rows = new Int32Array(s2Length)
  let state = -1

  for (let j = 0; j < s2Length; j++) {
    const index = s2Start + j
    const symbol = stringText ? s2.charCodeAt(index) : s2[index]
    const matches =
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
        ? direct[symbol - minSymbol]
        : wide === null
          ? 0
          : (wide.get(symbol) ?? 0)

    if (matches !== 0) {
      const u = state & matches
      state = (state + u) | 0 | (state & ~u)
    }

    rows[j] = state
  }

  return { sim: popcount32(~state & validBits(s1Length)), rows, words: 1 }
}
