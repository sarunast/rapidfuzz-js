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
let bandScores: Int32Array | null = null

export function directSlots(): Int32Array {
  return (asciiSlot ??= new Int32Array(directLimit))
}

export function directStamps(): Int32Array {
  return (asciiStamp ??= new Int32Array(directLimit))
}

export function maskPoolOf(): Int32Array {
  return (maskPool ??= new Int32Array(64))
}

/**
 * What one build hands its kernel: the generation the symbol table was stamped
 * with, the block masks the multi-word kernels index, and the lookup for
 * elements the direct table cannot hold.
 *
 * One per build, and never reused between builds — an escape-analysed object
 * measured flat against a mutable one held by the module (2026-08-16), which
 * is not worth a result whose fields the next build rewrites.
 *
 * A kernel reads its masks from here and never from `maskPoolOf()`. The two are
 * the same buffer for an ordinary build and are not for an oversized one, whose
 * symbol table holds offsets past everything the module kept — so reaching for
 * the module's pool would read a shorter buffer with another build's masks in
 * it. `wide` has the same rule for the opposite reason: nothing else holds it.
 */
export interface BuiltMasks {
  readonly stamp: number
  readonly pool: Int32Array
  readonly wide: ReadonlyMap<unknown, number>
}

const EMPTY_POOL = new Int32Array(0)
const EMPTY_WIDE: ReadonlyMap<unknown, number> = new Map<unknown, number>()

/** The longest pattern the mask memo will hold an entry for. */
const MASK_PATTERN_LIMIT = 4096

/**
 * What the pool would actually be allocated at to hold `needed` words, from
 * `size` up. A pool that is already larger doubles from where it is; the
 * retention cap below asks from the bottom, which is the same answer.
 */
function poolCapacity(needed: number, size = 64): number {
  while (size < needed) size *= 2
  return size
}

/**
 * The mask pool is the one buffer here that grows with `distinct * words`
 * rather than with the input length or the symbol space, so it is the one that
 * has to stop being the module's. Up to this many words it stays module-owned
 * and is reused; a build that needs more gets a buffer of its own, which the
 * kernel reads and then drops.
 *
 * Derived rather than chosen, because the two have to agree: a memo entry names
 * masks the module still holds, and the largest memoisable build is
 * `MASK_PATTERN_LIMIT` elements each taking a block of `words`. Cap it below
 * that and patterns the memo would hold start rebuilding — measured 1.22x on
 * repeated 4096-element comparisons over 2048 distinct elements (2026-08-16).
 *
 * Through `poolCapacity`, because what has to fit is the buffer the pool would
 * be given rather than the words a build asks for. The two coincide at 4096
 * and would not at, say, 5000, where the largest memoisable build asks for
 * 785,000 words and is handed 1,048,576.
 */
const RETAINED_MASK_WORDS = poolCapacity(
  MASK_PATTERN_LIMIT * wordCount(MASK_PATTERN_LIMIT),
)

export let directLimit: number = DIRECT_LOOKUP_LIMIT

let generation = 0

function grown(buffer: Int32Array | null, needed: number): Int32Array {
  if (buffer !== null && buffer.length >= needed) return buffer

  let size = buffer === null ? 64 : buffer.length
  while (size < needed) size *= 2
  return new Int32Array(size)
}

/**
 * Grows the mask pool, and decides who owns each size on the way: within the
 * cap it goes back to the module for the next build to reuse, past the cap it
 * does not, which is what leaves an oversized pool reachable only through the
 * `BuiltMasks` its build returned.
 *
 * Handing back each size on the way is a policy choice rather than a rule the
 * ownership needs: it decides what a build that then outgrows the cap leaves
 * behind, which is the largest capacity the cap allows rather than whatever the
 * build started from. It buys the next oversized build some doublings, and
 * those measured flat against publishing once at the end (2026-08-16) — a build
 * big enough to reach them is doing far more work in its own loop.
 */
function grownPool(pool: Int32Array, needed: number): Int32Array {
  if (pool.length >= needed) return pool

  const size = poolCapacity(needed, pool.length)
  const next = new Int32Array(size)
  next.set(pool)
  if (size <= RETAINED_MASK_WORDS) maskPool = next
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

export function buildWordMasks(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
): BuiltMasks {
  const stamp = nextGeneration()
  let slots = directSlots()
  let stamps = directStamps()
  let wide: Map<unknown, number> | null = null

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
      const held = (wide ??= new Map<unknown, number>())
      held.set(symbol, (held.get(symbol) ?? 0) | bit)
    }
  }

  return { stamp, pool: EMPTY_POOL, wide: wide ?? EMPTY_WIDE }
}

function buildBlockMasks(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
  words: number,
): BuiltMasks {
  const stamp = nextGeneration()
  let slots = directSlots()
  let stamps = directStamps()
  let wide: Map<unknown, number> | null = null

  let pool = maskPoolOf()
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
        pool = grownPool(pool, offset + words)
        clearRange(pool, 0, offset, offset + words)
        slots[symbol] = offset
        stamps[symbol] = stamp
        distinct++
      }

      const word = offset + (i >>> WORD_SHIFT)
      pool[word] = pool[word] | (1 << (i & WORD_MASK))
    }

    return { stamp, pool, wide: EMPTY_WIDE }
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
      : wide === null
        ? -1
        : (wide.get(symbol) ?? -1)

    if (offset < 0) {
      offset = distinct * words
      pool = grownPool(pool, offset + words)
      clearRange(pool, 0, offset, offset + words)

      if (direct) {
        slots[symbol] = offset
        stamps[symbol] = stamp
      } else {
        const held = (wide ??= new Map<unknown, number>())
        held.set(symbol, offset)
      }

      distinct++
    }

    const word = offset + (i >>> WORD_SHIFT)
    pool[word] = pool[word] | (1 << (i & WORD_MASK))
  }

  return { stamp, pool, wide: wide ?? EMPTY_WIDE }
}

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
): BuiltMasks {
  if (
    typeof pattern !== 'string' ||
    maskPattern !== pattern ||
    maskStart !== start ||
    maskLength !== length ||
    maskWords !== words ||
    maskGeneration !== generation
  ) {
    const masks = buildBlockMasks(pattern, start, length, words)
    // A memo entry names masks the module still holds, so a build that kept its
    // pool to itself cannot leave one behind: `maskPool` is another build's.
    const retained = masks.pool === maskPool
    maskPattern =
      retained && typeof pattern === 'string' && pattern.length <= MASK_PATTERN_LIMIT
        ? pattern
        : null
    maskStart = start
    maskLength = length
    maskWords = words
    maskGeneration = masks.stamp
    return masks
  }

  return { stamp: maskGeneration, pool: maskPoolOf(), wide: EMPTY_WIDE }
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
  bandScores = null
  directLimit = DIRECT_LOOKUP_LIMIT
  invalidateMaskCache()
}
