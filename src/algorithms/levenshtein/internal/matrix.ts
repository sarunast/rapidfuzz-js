import { wordCount } from '../../bitmask/blockMasks.js'
import {
  maskBlockBound,
  oneWordMasks,
  symbolSpan,
  wordPositionMasks,
} from '../../bitmask/positionMasks.js'

interface LevenshteinMatrix {
  readonly dist: number
  readonly vp: Int32Array
  readonly vn: Int32Array
  readonly stride: number
  readonly offsets: Int32Array | null
}

function bandedRows(s1Length: number, maximumDistance: number): boolean {
  return s1Length > 32 && maximumDistance >= 0 && 2 * maximumDistance + 1 < s1Length
}

function rowStride(s1Length: number, maximumDistance: number): number {
  const words = wordCount(s1Length)
  return bandedRows(s1Length, maximumDistance)
    ? Math.min(words, Math.floor((2 * maximumDistance + 63) / 32))
    : words
}

export function levenshteinRowBytes(
  s1Length: number,
  s2Length: number,
  maximumDistance: number,
): number {
  const banded = bandedRows(s1Length, maximumDistance)
  return s2Length * (2 * rowStride(s1Length, maximumDistance) * 4 + (banded ? 4 : 0))
}

export function levenshteinMatrixBytes(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2Length: number,
  maximumDistance: number,
): number {
  const rows = levenshteinRowBytes(s1Length, s2Length, maximumDistance)
  const words = wordCount(s1Length)
  if (s2Length === 0 || words < 2) return rows

  const span = symbolSpan(s1, s1Start, s1Length)
  const { minSymbol, maxSymbol } = span
  const bases = minSymbol > maxSymbol ? 0 : maxSymbol - minSymbol + 1
  return rows + (maskBlockBound(span, s1Length) * words + bases + 2 * words) * 4
}

export function levenshteinMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
  maximumDistance = -1,
): LevenshteinMatrix {
  if (s1Length === 0 || s2Length === 0) {
    return {
      dist: s1Length + s2Length,
      vp: new Int32Array(0),
      vn: new Int32Array(0),
      stride: 0,
      offsets: null,
    }
  }

  const words = wordCount(s1Length)
  if (words === 1) {
    return oneWordLevenshteinMatrix(s1, s1Start, s1Length, s2, s2Start, s2Length)
  }

  const banded = bandedRows(s1Length, maximumDistance)
  const stride = rowStride(s1Length, maximumDistance)
  const offsets = banded ? new Int32Array(s2Length) : null

  const { masks, bases, minSymbol, maxSymbol, wide } = wordPositionMasks(
    s1,
    s1Start,
    s1Length,
    words,
  )
  const stringText = typeof s2 === 'string'
  const vpState = new Int32Array(words).fill(-1)
  const vnState = new Int32Array(words)
  const vp = new Int32Array(s2Length * stride)
  const vn = new Int32Array(s2Length * stride)

  const lastWord = words - 1
  const top = 1 << ((s1Length - 1) & 31)
  let currDist = s1Length

  for (let j = 0; j < s2Length; j++) {
    const index = s2Start + j
    const symbol = stringText ? s2.charCodeAt(index) : s2[index]
    const matchBase =
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
        ? bases[symbol - minSymbol]
        : wide === null
          ? 0
          : (wide.get(symbol) ?? 0)
    let addCarry = 0
    let carryP = 1
    let carryN = 0

    for (let w = 0; w < words; w++) {
      const vpWord = vpState[w]
      const vnWord = vnState[w]
      const x = masks[matchBase + w]

      const addend = x & vpWord
      const sum = (addend + vpWord + addCarry) | 0
      addCarry = ((addend & vpWord) | ((addend | vpWord) & ~sum)) >>> 31

      const d0 = (sum ^ vpWord) | x | vnWord
      const hp = vnWord | ~(d0 | vpWord)
      const hn = d0 & vpWord

      if (w === lastWord) {
        if ((hp & top) !== 0) currDist++
        if ((hn & top) !== 0) currDist--
      }

      const shiftedP = (hp << 1) | carryP
      const shiftedN = (hn << 1) | carryN
      carryP = hp >>> 31
      carryN = hn >>> 31

      vpState[w] = shiftedN | ~(d0 | shiftedP)
      vnState[w] = shiftedP & d0
    }

    const base = j * stride
    if (offsets === null) {
      vp.set(vpState, base)
      vn.set(vnState, base)
      continue
    }

    const from = Math.min(words - stride, Math.max(0, j + 1 - maximumDistance - 1) >>> 5)
    offsets[j] = from << 5
    for (let w = 0; w < stride; w++) {
      vp[base + w] = vpState[from + w]
      vn[base + w] = vnState[from + w]
    }
  }

  return { dist: currDist, vp, vn, stride, offsets }
}

function oneWordLevenshteinMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
): LevenshteinMatrix {
  const { direct, minSymbol, maxSymbol, wide } = oneWordMasks(s1, s1Start, s1Length)
  const stringText = typeof s2 === 'string'
  const vp = new Int32Array(s2Length)
  const vn = new Int32Array(s2Length)

  const top = 1 << (s1Length - 1)
  let vpState = -1
  let vnState = 0
  let currDist = s1Length

  for (let j = 0; j < s2Length; j++) {
    const index = s2Start + j
    const symbol = stringText ? s2.charCodeAt(index) : s2[index]
    const x =
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
        ? direct[symbol - minSymbol]
        : wide === null
          ? 0
          : (wide.get(symbol) ?? 0)

    const addend = x & vpState
    const sum = (addend + vpState) | 0

    const d0 = (sum ^ vpState) | x | vnState
    const hp = vnState | ~(d0 | vpState)
    const hn = d0 & vpState

    if ((hp & top) !== 0) currDist++
    if ((hn & top) !== 0) currDist--

    const shiftedP = (hp << 1) | 1
    const shiftedN = hn << 1

    vpState = shiftedN | ~(d0 | shiftedP)
    vnState = shiftedP & d0
    vp[j] = vpState
    vn[j] = vnState
  }

  return { dist: currDist, vp, vn, stride: 1, offsets: null }
}
