// Private to preparation. `blockMasks.ts` partitions symbols differently and
// keeps its own copy of the limit, which folds where a cross-module binding
// does not — the two are not one rule with two call sites.
const DIRECT_LOOKUP_LIMIT = 256

function isDirectSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  )
}

/**
 * An integer at or above the direct table, which is the candidate set for the
 * dense high window — not a guarantee of a place in it, since a symbol outside
 * the chosen span ends up in `wideOffsets` instead.
 */
function isHighIntegerSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' && symbol >= DIRECT_LOOKUP_LIMIT && (symbol | 0) === symbol
  )
}

const WORD_SHIFT = 5
const WORD_MASK = 31

const WINDOW_SPAN_LIMIT = 2048

const WINDOW_CELL_LIMIT = 16_384

const EMPTY_WIDE: ReadonlyMap<unknown, number> = new Map<unknown, number>()

const EMPTY_MASKS = new Int32Array(0)
const EMPTY_PATTERN: PatternMask = /* @__PURE__ */ Object.freeze({
  length: 0,
  words: 0,
  masks: EMPTY_MASKS,
  highBase: 0,
  highCount: 0,
  highStart: 0,
  wideOffsets: EMPTY_WIDE,
})

export interface PatternMask {
  readonly length: number
  readonly words: number
  readonly masks: Int32Array
  readonly highBase: number
  readonly highCount: number
  readonly highStart: number
  readonly wideOffsets: ReadonlyMap<unknown, number>
}

export function preparePattern(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
  step = 1,
): PatternMask {
  if (length === 0) return EMPTY_PATTERN

  const words = ((length - 1) >>> WORD_SHIFT) + 1
  const stringPattern = typeof pattern === 'string'
  const directCells = DIRECT_LOOKUP_LIMIT * words

  let masks = new Int32Array(directCells)
  let low = 0
  let high = -1
  let highs: { at: number[]; of: number[] } | null = null
  let strays: { at: number[]; slot: number[]; index: Map<unknown, number> } | null = null

  for (let i = 0; i < length; i++) {
    const index = start + i * step
    const symbol = stringPattern ? pattern.charCodeAt(index) : pattern[index]
    if (isDirectSymbol(symbol)) {
      masks[symbol * words + (i >>> WORD_SHIFT)] |= 1 << (i & WORD_MASK)
      continue
    }
    if (symbol !== symbol) continue
    if (isHighIntegerSymbol(symbol)) {
      highs ??= { at: [], of: [] }
      highs.at.push(i)
      highs.of.push(symbol)
      if (high < 0 || symbol < low) low = symbol
      if (symbol > high) high = symbol
      continue
    }
    strays ??= { at: [], slot: [], index: new Map<unknown, number>() }
    let slot = strays.index.get(symbol)
    if (slot === undefined) {
      slot = strays.index.size
      strays.index.set(symbol, slot)
    }
    strays.at.push(i)
    strays.slot.push(slot)
  }

  if (highs === null && strays === null) {
    return Object.freeze({
      length,
      words,
      masks,
      highBase: 0,
      highCount: 0,
      highStart: directCells,
      wideOffsets: EMPTY_WIDE,
    })
  }

  const span = high < 0 ? 0 : high - low + 1
  const windowed =
    span > 0 && span <= WINDOW_SPAN_LIMIT && span * words <= WINDOW_CELL_LIMIT
  const highBase = windowed ? low : 0
  const highCount = windowed ? span : 0

  if (!windowed && highs !== null) {
    const moved = (strays ??= { at: [], slot: [], index: new Map<unknown, number>() })
    for (let d = 0; d < highs.at.length; d++) {
      const symbol = highs.of[d]
      let slot = moved.index.get(symbol)
      if (slot === undefined) {
        slot = moved.index.size
        moved.index.set(symbol, slot)
      }
      moved.at.push(highs.at[d])
      moved.slot.push(slot)
    }
    highs = null
  }

  const highStart = directCells
  const strayStart = highStart + highCount * words
  const strayCount = strays === null ? 0 : strays.index.size

  const grown = new Int32Array(strayStart + strayCount * words)
  grown.set(masks)
  masks = grown

  if (highs !== null) {
    for (let d = 0; d < highs.at.length; d++) {
      const i = highs.at[d]
      const base = highStart + (highs.of[d] - highBase) * words
      masks[base + (i >>> WORD_SHIFT)] |= 1 << (i & WORD_MASK)
    }
  }

  let wideOffsets: ReadonlyMap<unknown, number> = EMPTY_WIDE
  if (strays !== null) {
    for (let d = 0; d < strays.at.length; d++) {
      const i = strays.at[d]
      const base = strayStart + strays.slot[d] * words
      masks[base + (i >>> WORD_SHIFT)] |= 1 << (i & WORD_MASK)
    }
    for (const [symbol, slot] of strays.index) {
      strays.index.set(symbol, strayStart + slot * words)
    }
    wideOffsets = strays.index
  }

  return Object.freeze({
    length,
    words,
    masks,
    highBase,
    highCount,
    highStart,
    wideOffsets,
  })
}
