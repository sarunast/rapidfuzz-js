export const WORD_BITS = 32
export const WORD_SHIFT = 5
export const WORD_MASK = 31

const DIRECT_LOOKUP_LIMIT = 256

export function wordCount(length: number): number {
  return length === 0 ? 0 : ((length - 1) >>> WORD_SHIFT) + 1
}

const DIRECT_LOOKUP_MAX = 0x1_0000

let maskPool: Int32Array | null = null
let vectorP: Int32Array | null = null
let vectorN: Int32Array | null = null
let asciiSlot: Int32Array | null = null
let asciiStamp: Int32Array | null = null
let wideSlot: Map<unknown, number> | null = null
let bandScores: Int32Array | null = null

export function directSlots(): Int32Array {
  return (asciiSlot ??= new Int32Array(directLimit))
}

export function directStamps(): Int32Array {
  return (asciiStamp ??= new Int32Array(directLimit))
}

export function wideSlots(): Map<unknown, number> {
  return (wideSlot ??= new Map<unknown, number>())
}

export function maskPoolOf(): Int32Array {
  return (maskPool ??= new Int32Array(64))
}

const RETAINED_MASK_WORDS = 1 << 18

function maskPoolFor(needed: number): Int32Array {
  const pool = maskPoolOf()
  if (pool.length <= RETAINED_MASK_WORDS || needed > RETAINED_MASK_WORDS) return pool

  return (maskPool = new Int32Array(64))
}

function dropOversizedMaskPool(): void {
  if (maskPool !== null && maskPool.length > RETAINED_MASK_WORDS) maskPool = null
}

export let directLimit: number = DIRECT_LOOKUP_LIMIT

let generation = 0

function grown(buffer: Int32Array | null, needed: number): Int32Array {
  if (buffer !== null && buffer.length >= needed) return buffer

  let size = buffer === null ? 64 : buffer.length
  while (size < needed) size *= 2
  return new Int32Array(size)
}

function grownPreserving(buffer: Int32Array, needed: number): Int32Array {
  if (buffer.length >= needed) return buffer

  let size = buffer.length
  while (size < needed) size *= 2
  const next = new Int32Array(size)
  next.set(buffer)
  return next
}

export function clearRange(
  buffer: Int32Array,
  value: number,
  start: number,
  end: number,
): void {
  if (end - start >= 64) {
    buffer.fill(value, start, end)
    return
  }
  for (let i = start; i < end; i++) buffer[i] = value
}

function widenDirect(
  symbol: number,
  heldSlots: Int32Array,
  heldStamps: Int32Array,
): [Int32Array, Int32Array] {
  let size = directLimit
  while (size <= symbol) size *= 2

  const slots = new Int32Array(size)
  const stamps = new Int32Array(size)
  slots.set(heldSlots)
  stamps.set(heldStamps)

  asciiSlot = slots
  asciiStamp = stamps
  directLimit = size
  return [slots, stamps]
}

const GENERATION_LIMIT = 0x7fff_ffff

function nextGeneration(): number {
  generation++
  if (generation >= GENERATION_LIMIT) {
    directStamps().fill(0)
    invalidateMaskCache()
    generation = 1
  }

  return generation
}

export function checkedStartGeneration(startGeneration: number): number {
  if (
    !Number.isInteger(startGeneration) ||
    startGeneration < 0 ||
    startGeneration >= GENERATION_LIMIT
  ) {
    throw new RangeError(
      `startGeneration has to be an integer in 0 - ${GENERATION_LIMIT - 1}`,
    )
  }

  return startGeneration
}

function clearWide(wide: Map<unknown, number>): void {
  if (wide.size !== 0) wide.clear()
}

export function buildWordMasks(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
): number {
  const stamp = nextGeneration()
  let slots = directSlots()
  let stamps = directStamps()
  const wide = wideSlots()

  clearWide(wide)
  dropOversizedMaskPool()
  let limit = directLimit
  const stringPattern = typeof pattern === 'string'

  for (let i = 0; i < length; i++) {
    const symbol = stringPattern ? pattern.charCodeAt(start + i) : pattern[start + i]
    const bit = 1 << i

    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      slots[symbol] = (stamps[symbol] === stamp ? slots[symbol] : 0) | bit
      stamps[symbol] = stamp
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < DIRECT_LOOKUP_MAX &&
      (symbol | 0) === symbol
    ) {
      if (symbol >= limit) {
        const widened = widenDirect(symbol, slots, stamps)
        slots = widened[0]
        stamps = widened[1]
        limit = directLimit
      }

      slots[symbol] = (stamps[symbol] === stamp ? slots[symbol] : 0) | bit
      stamps[symbol] = stamp
    } else if (symbol === symbol) {
      wide.set(symbol, (wide.get(symbol) ?? 0) | bit)
    }
  }

  return stamp
}

function buildBlockMasks(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
  words: number,
): number {
  const stamp = nextGeneration()
  let slots = directSlots()
  let stamps = directStamps()
  const wide = wideSlots()

  clearWide(wide)

  let pool = maskPoolFor(length * words)
  let distinct = 0
  let limit = directLimit
  const stringPattern = typeof pattern === 'string'

  if (stringPattern) {
    for (let i = 0; i < length; i++) {
      const symbol = pattern.charCodeAt(start + i)
      if (symbol >= limit) {
        const widened = widenDirect(symbol, slots, stamps)
        slots = widened[0]
        stamps = widened[1]
        limit = directLimit
      }

      let offset = stamps[symbol] === stamp ? slots[symbol] : -1
      if (offset < 0) {
        offset = distinct * words
        pool = grownPreserving(pool, offset + words)
        maskPool = pool
        clearRange(pool, 0, offset, offset + words)
        slots[symbol] = offset
        stamps[symbol] = stamp
        distinct++
      }

      const word = offset + (i >>> WORD_SHIFT)
      pool[word] = pool[word] | (1 << (i & WORD_MASK))
    }

    return stamp
  }

  for (let i = 0; i < length; i++) {
    const symbol = pattern[start + i]
    const direct =
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_MAX &&
      (symbol | 0) === symbol

    if (direct && symbol >= limit) {
      const widened = widenDirect(symbol, slots, stamps)
      slots = widened[0]
      stamps = widened[1]
      limit = directLimit
    }

    if (!direct && symbol !== symbol) continue

    let offset = direct
      ? stamps[symbol] === stamp
        ? slots[symbol]
        : -1
      : (wide.get(symbol) ?? -1)

    if (offset < 0) {
      offset = distinct * words
      pool = grownPreserving(pool, offset + words)
      maskPool = pool
      clearRange(pool, 0, offset, offset + words)

      if (direct) {
        slots[symbol] = offset
        stamps[symbol] = stamp
      } else {
        wide.set(symbol, offset)
      }

      distinct++
    }

    const word = offset + (i >>> WORD_SHIFT)
    pool[word] = pool[word] | (1 << (i & WORD_MASK))
  }

  return stamp
}

const MASK_PATTERN_LIMIT = 4096
let maskPattern: string | null = null
let maskStart = -1
let maskLength = -1
let maskWords = -1
let maskGeneration = 0

function invalidateMaskCache(): void {
  maskPattern = null
  maskStart = -1
  maskLength = -1
  maskWords = -1
  maskGeneration = 0
}

export function blockMasksFor(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
  words: number,
): number {
  if (
    typeof pattern !== 'string' ||
    maskPattern !== pattern ||
    maskStart !== start ||
    maskLength !== length ||
    maskWords !== words ||
    maskGeneration !== generation
  ) {
    const stamp = buildBlockMasks(pattern, start, length, words)
    maskPattern =
      typeof pattern === 'string' && pattern.length <= MASK_PATTERN_LIMIT ? pattern : null
    maskStart = start
    maskLength = length
    maskWords = words
    maskGeneration = stamp
    return stamp
  }

  return maskGeneration
}

export const WORD_LIMIT: number = WORD_BITS

export const UNBOUNDED_MISSES: number = Number.MAX_SAFE_INTEGER

export let affixPrefix = 0
export let affixLen1 = 0
export let affixLen2 = 0

export function measureAffix(
  s1: ArrayLike<unknown>,
  start1: number,
  len1: number,
  s2: ArrayLike<unknown>,
  start2: number,
  len2: number,
): void {
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    measureAffixString(s1, start1, len1, s2, start2, len2)
    return
  }

  const shorter = len1 < len2 ? len1 : len2

  let prefix = 0
  while (prefix < shorter && s1[start1 + prefix] === s2[start2 + prefix]) prefix++

  let suffix = 0
  while (
    suffix < shorter - prefix &&
    s1[start1 + len1 - suffix - 1] === s2[start2 + len2 - suffix - 1]
  ) {
    suffix++
  }

  affixPrefix = prefix
  affixLen1 = len1 - prefix - suffix
  affixLen2 = len2 - prefix - suffix
}

function measureAffixString(
  s1: string,
  start1: number,
  len1: number,
  s2: string,
  start2: number,
  len2: number,
): void {
  const shorter = len1 < len2 ? len1 : len2

  let prefix = 0
  while (
    prefix < shorter &&
    s1.charCodeAt(start1 + prefix) === s2.charCodeAt(start2 + prefix)
  ) {
    prefix++
  }

  let suffix = 0
  while (
    suffix < shorter - prefix &&
    s1.charCodeAt(start1 + len1 - suffix - 1) ===
      s2.charCodeAt(start2 + len2 - suffix - 1)
  ) {
    suffix++
  }

  affixPrefix = prefix
  affixLen1 = len1 - prefix - suffix
  affixLen2 = len2 - prefix - suffix
}

export function rowVector(words: number): Int32Array {
  vectorP = grown(vectorP, words)
  return vectorP
}

export function rowVectorN(words: number): Int32Array {
  vectorN = grown(vectorN, words)
  return vectorN
}

export function bandVector(words: number): Int32Array {
  bandScores = grown(bandScores, words)
  return bandScores
}

export function resetBitVectorScratch(startGeneration = 0): void {
  generation = checkedStartGeneration(startGeneration)
  maskPool = null
  vectorP = null
  vectorN = null
  asciiSlot = null
  asciiStamp = null
  wideSlot = null
  bandScores = null
  directLimit = DIRECT_LOOKUP_LIMIT
  invalidateMaskCache()
}
