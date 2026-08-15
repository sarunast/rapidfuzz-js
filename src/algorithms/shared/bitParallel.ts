function wordCount(length: number): number {
  return length === 0 ? 0 : ((length - 1) >>> 5) + 1
}

function popcount32(word: number): number {
  let bits = word - ((word >>> 1) & 0x5555_5555)
  bits = (bits & 0x3333_3333) + ((bits >>> 2) & 0x3333_3333)
  return (((bits + (bits >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24
}

const SPAN_SLACK = 256

const NO_MASKS = new Int32Array(0)

interface SymbolSpan {
  readonly minSymbol: number
  readonly maxSymbol: number
}

interface SymbolCoverage extends SymbolSpan {
  readonly spans: boolean
}

const EMPTY_SPAN: SymbolCoverage = { minSymbol: 1, maxSymbol: 0, spans: false }

function symbolSpan(
  s1: ArrayLike<unknown>,
  start: number,
  length: number,
): SymbolCoverage {
  const stringPattern = typeof s1 === 'string'
  let minSymbol = 1
  let maxSymbol = 0
  let spans = true

  for (let i = 0; i < length; i++) {
    const index = start + i
    const symbol = stringPattern ? s1.charCodeAt(index) : s1[index]
    if (typeof symbol !== 'number' || (symbol | 0) !== symbol) {
      spans = false
      continue
    }

    if (minSymbol > maxSymbol) {
      minSymbol = symbol
      maxSymbol = symbol
    } else if (symbol < minSymbol) minSymbol = symbol
    else if (symbol > maxSymbol) maxSymbol = symbol
  }

  return maxSymbol - minSymbol >= length + SPAN_SLACK
    ? EMPTY_SPAN
    : { minSymbol, maxSymbol, spans }
}

function maskBlockBound(span: SymbolCoverage, length: number): number {
  const { minSymbol, maxSymbol, spans } = span
  return spans && minSymbol <= maxSymbol
    ? Math.min(maxSymbol - minSymbol + 1, length) + 1
    : length + 1
}

interface OneWordMasks extends SymbolSpan {
  readonly direct: Int32Array
  readonly wide: ReadonlyMap<unknown, number> | null
}

function oneWordMasks(
  s1: ArrayLike<unknown>,
  start: number,
  length: number,
): OneWordMasks {
  const { minSymbol, maxSymbol } = symbolSpan(s1, start, length)
  const direct =
    minSymbol > maxSymbol ? NO_MASKS : new Int32Array(maxSymbol - minSymbol + 1)
  const stringPattern = typeof s1 === 'string'
  let wide: Map<unknown, number> | null = null

  for (let i = 0; i < length; i++) {
    const index = start + i
    const symbol = stringPattern ? s1.charCodeAt(index) : s1[index]
    const bit = 1 << i

    if (
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
    ) {
      direct[symbol - minSymbol] |= bit
      continue
    }

    if (symbol !== symbol) continue
    if (wide === null) wide = new Map<unknown, number>()
    wide.set(symbol, (wide.get(symbol) ?? 0) | bit)
  }

  return { direct, minSymbol, maxSymbol, wide }
}

interface WordMasks extends SymbolSpan {
  readonly masks: Int32Array
  readonly bases: Int32Array
  readonly wide: ReadonlyMap<unknown, number> | null
}

const INITIAL_BLOCKS = 64

const MAX_SPAN_BLOCKS = 256

function grownMasks(
  masks: Int32Array,
  blocks: number,
  words: number,
  spanSize: number,
  limit: number,
): Int32Array {
  let size = Math.max(masks.length, spanSize)
  while (size < blocks * words) size *= 2

  const grown = new Int32Array(Math.min(size, limit))
  grown.set(masks)
  return grown
}

function wordPositionMasks(
  s1: ArrayLike<unknown>,
  start: number,
  length: number,
  words: number,
): WordMasks {
  const { minSymbol, maxSymbol } = symbolSpan(s1, start, length)
  const bases =
    minSymbol > maxSymbol ? NO_MASKS : new Int32Array(maxSymbol - minSymbol + 1)
  const stringPattern = typeof s1 === 'string'
  let wide: Map<unknown, number> | null = null
  const limit = (length + 1) * words
  const spanSize =
    minSymbol > maxSymbol ? 0 : (Math.min(maxSymbol - minSymbol + 1, length) + 1) * words
  let masks: Int32Array = new Int32Array(
    spanSize === 0
      ? Math.min(limit, (INITIAL_BLOCKS + 1) * words)
      : Math.min(spanSize, (MAX_SPAN_BLOCKS + 1) * words),
  )
  let blocks = 1

  for (let i = 0; i < length; i++) {
    const index = start + i
    const symbol = stringPattern ? s1.charCodeAt(index) : s1[index]
    let base: number

    if (
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
    ) {
      const entry = symbol - minSymbol
      base = bases[entry]
      if (base === 0) {
        base = blocks++ * words
        if (base + words > masks.length)
          masks = grownMasks(masks, blocks, words, spanSize, limit)
        bases[entry] = base
      }
    } else {
      if (symbol !== symbol) continue
      if (wide === null) wide = new Map<unknown, number>()
      const held = wide.get(symbol)
      if (held === undefined) {
        base = blocks++ * words
        if (base + words > masks.length)
          masks = grownMasks(masks, blocks, words, spanSize, limit)
        wide.set(symbol, base)
      } else {
        base = held
      }
    }

    masks[base + (i >>> 5)] |= 1 << (i & 31)
  }

  return { masks, bases, minSymbol, maxSymbol, wide }
}

function validBits(length: number): number {
  const bits = length & 31
  return bits === 0 ? -1 : ~(-1 << bits)
}

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

export function rowBitSet(
  rows: Int32Array,
  words: number,
  row: number,
  pos: number,
): boolean {
  return (rows[row * words + (pos >>> 5)] & (1 << (pos & 31))) !== 0
}

export function shiftedRowBitSet(
  rows: Int32Array,
  stride: number,
  row: number,
  offset: number,
  pos: number,
): boolean {
  const relative = pos - offset
  if (relative < 0 || relative >= stride << 5) return false
  return (rows[row * stride + (relative >>> 5)] & (1 << (relative & 31))) !== 0
}
