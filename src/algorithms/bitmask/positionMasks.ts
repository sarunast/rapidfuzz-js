// Freshly allocated, deliberately, where `blockMasks.ts` pools and stamps.
// The alignment path retains the row matrix built from these past the call
// that produced it, so a pooled buffer would be overwritten under a caller
// still reading it. The two therefore look alike and must not be merged.

const SPAN_SLACK = 256

const NO_MASKS = new Int32Array(0)

interface SymbolSpan {
  readonly minSymbol: number
  readonly maxSymbol: number
}

export interface SymbolCoverage extends SymbolSpan {
  readonly spans: boolean
}

const EMPTY_SPAN: SymbolCoverage = { minSymbol: 1, maxSymbol: 0, spans: false }

export function popcount32(word: number): number {
  let bits = word - ((word >>> 1) & 0x5555_5555)
  bits = (bits & 0x3333_3333) + ((bits >>> 2) & 0x3333_3333)
  return (((bits + (bits >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24
}

export function validBits(length: number): number {
  const bits = length & 31
  return bits === 0 ? -1 : ~(-1 << bits)
}

export function symbolSpan(
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

export function maskBlockBound(span: SymbolCoverage, length: number): number {
  const { minSymbol, maxSymbol, spans } = span
  return spans && minSymbol <= maxSymbol
    ? Math.min(maxSymbol - minSymbol + 1, length) + 1
    : length + 1
}

export interface OneWordMasks extends SymbolSpan {
  readonly direct: Int32Array
  readonly wide: ReadonlyMap<unknown, number> | null
}

export function oneWordMasks(
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

export interface WordMasks extends SymbolSpan {
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

export function wordPositionMasks(
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
