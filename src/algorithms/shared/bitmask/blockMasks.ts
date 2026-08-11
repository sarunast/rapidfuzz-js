/**
 * Shared lookup tables, block-mask construction, and scratch storage.
 *
 * The kernels in `lcs.ts`, `levenshtein.ts` and `osa.ts` allocate nothing per
 * call: they reuse the module-level buffers below, grown on demand. This is
 * safe because no kernel calls back into user code — no normalizer, no scorer,
 * no iterator runs between acquiring a buffer and releasing it — so a call can
 * never observe another call's buffer. Buffers are created lazily rather than
 * at module scope, because `"sideEffects": false` promises that importing this
 * module does no work.
 *
 * ## Reading this state from another module
 *
 * The buffers and `directLimit` are exported as live bindings, and every kernel
 * hoists the ones it needs into locals *once per call* before its loop starts.
 * That is what makes the split across modules free: per-element work touches a
 * local, never an import.
 *
 * The row vectors are the exception. A kernel used to grow them by assigning
 * the module variable, which is not something an importer can do, so they are
 * reached through {@link rowVector}, {@link rowVectorN} and
 * {@link bandVector} instead — one call per kernel invocation.
 *
 * ## The hot constants are duplicated, not imported
 *
 * `WORD_BITS`, `WORD_SHIFT`, `WORD_MASK`, `DIRECT_LOOKUP_LIMIT` and
 * `DIRECT_LOOKUP_MAX` are exported here as the canonical definitions, and each
 * kernel module declares its own copies. They are read once per element, where
 * a cross-module binding does not fold the way a module-local `const` does —
 * measured at +3% on Latin-1 and +15% on Cyrillic. Any copy that disagrees with
 * the values here is a bug; the partition invariant below is what it breaks.
 */

/** Bits per word. The shift/mask constants below assume 32 and are not general. */
export const WORD_BITS = 32
export const WORD_SHIFT = 5
export const WORD_MASK = 31

/**
 * Elements below this index a plain array instead of a `Map`. Latin-1 is
 * covered rather than just ASCII, which is what upstream's
 * `PatternMatchVector` does and costs nothing but table size.
 *
 * This is only where the shared table starts — see {@link directLimit}. It
 * remains the fixed width of a {@link PatternMask}, which is allocated per
 * prepared query rather than once for the process.
 */
export const DIRECT_LOOKUP_LIMIT = 256

/**
 * Widest the shared table grows: the whole Basic Multilingual Plane.
 *
 * Strings reach the kernels either as `charCodeAt` reads, which stop at
 * `0xffff`, or as code points converted by `convPair` — and that conversion only
 * happens for text that has a surrogate pair, so astral code points above this
 * are the one string case that still takes the `Map`. They are rare enough to
 * be worth the four megabytes not spent covering them.
 */
export const DIRECT_LOOKUP_MAX = 0x1_0000

/**
 * ## Which elements the direct lookup table can hold
 *
 * Every kernel below decides that inline, with the same four-part test:
 *
 * ```
 * typeof symbol === 'number' && symbol >= 0 &&
 *   symbol < DIRECT_LOOKUP_LIMIT && (symbol | 0) === symbol
 * ```
 *
 * followed by the same test again against `[DIRECT_LOOKUP_LIMIT, limit)`, which
 * is the part of the table {@link widenDirect} has added. Those two could be
 * one test against `limit`, and were: a variable bound in place of the constant
 * cost between three and seventeen percent on Latin-1 text, which is the input
 * that never leaves the first range. Splitting them leaves that path reading
 * the constant it always read, and puts the widened range on a branch only
 * higher text reaches.
 *
 * The integer part is load bearing. A typed array has no element at a
 * fractional index, so `slots[1.5] = mask` is silently dropped and reads back
 * as `undefined` — without it, `1.5` would match nothing, not even itself. The
 * limit never exceeds `0x10000`, so `| 0` is a sound integer test under it.
 *
 * Everything else goes to `wideSlot`, except `NaN`. A `Map` matches keys by
 * SameValueZero, under which `NaN` equals itself; every other comparison in this
 * library is `===`, under which it does not — including the affix trimming below
 * and the scalar fallbacks in `osa.ts` and `damerauLevenshtein.ts`. Each site
 * therefore spells `NaN` as `symbol !== symbol` and gives it no match bit, so
 * the answer cannot depend on which path ran.
 *
 * The test is written out at each site rather than called: it runs once per
 * element of the streamed input, where a call the optimiser declines to inline
 * showed up as several percent of total time.
 *
 * ## The limit has to be the same on both sides
 *
 * A mask builder and the kernel that reads its masks partition the symbols
 * between the table and the `Map` by comparing against `limit`. Read a symbol
 * under a wider limit than it was stored under and the table is consulted for
 * something the `Map` holds, which reports "not in the pattern" for an element
 * that is in it. So every kernel hoists {@link directLimit} *after* building,
 * and a builder that widens mid-pattern rereads it — see {@link widenDirect}.
 */

let maskPool: Int32Array | null = null
let vectorP: Int32Array | null = null
let vectorN: Int32Array | null = null
let asciiSlot: Int32Array | null = null
let asciiStamp: Int32Array | null = null
let wideSlot: Map<unknown, number> | null = null
let bandScores: Int32Array | null = null

/**
 * The shared table's two halves, its overflow map and the mask pool, allocated
 * on first use.
 *
 * Reached through these rather than through the bindings themselves. A kernel
 * that hoisted a binding held an `Int32Array | null` and had to re-establish
 * what an earlier call had already guaranteed, and the only way to do that was
 * a null test on a path nothing could reach — one per kernel, sixteen of them,
 * each one a branch no test could ever cover. Returning the buffer proves it
 * instead, and costs one call per kernel invocation rather than per element.
 *
 * The first-use allocation is what keeps the promise `"sideEffects": false`
 * makes: importing this module still does no work.
 */
export function directSlots(): Int32Array {
  return (asciiSlot ??= new Int32Array(directLimit))
}

/** The stamp half of the shared table — see {@link directSlots}. */
export function directStamps(): Int32Array {
  return (asciiStamp ??= new Int32Array(directLimit))
}

/** Masks for elements the shared table cannot index — see {@link directSlots}. */
export function wideSlots(): Map<unknown, number> {
  return (wideSlot ??= new Map<unknown, number>())
}

/**
 * The multi-word mask pool — see {@link directSlots}.
 *
 * Born at 64 words rather than empty so that {@link grownPreserving} always has
 * a size to double.
 */
export function maskPoolOf(): Int32Array {
  return (maskPool ??= new Int32Array(64))
}

/**
 * How much of the symbol space the shared table currently covers.
 *
 * Text outside Latin-1 — Greek, Cyrillic, Hebrew, CJK, anything at all in most
 * of the world's scripts — took the `Map` at every element, which measured
 * around six times the cost of the same work in ASCII and gave up the kernels'
 * whole advantage over a plain 32-bit Myers implementation on long inputs.
 *
 * The table grows to cover what it is actually asked for rather than being
 * born at {@link DIRECT_LOOKUP_MAX}: a process that only ever scores ASCII
 * keeps two kilobytes, one that scores Cyrillic ends at sixteen, and only text
 * high in the BMP pays the full half megabyte. Growth is permanent, and it is
 * bounded by the eight doublings between the two constants.
 */
export let directLimit: number = DIRECT_LOOKUP_LIMIT

/**
 * Generation counter for the ASCII table. Stamping a slot with the current call
 * lets the table be reused without being cleared, which matters because
 * clearing the table would dominate the cost of scoring a short string.
 */
let generation = 0

/**
 * Grow a scratch buffer, discarding whatever was in it.
 *
 * Only safe for the row vectors, which are refilled before use.
 */
function grown(buffer: Int32Array | null, needed: number): Int32Array {
  if (buffer !== null && buffer.length >= needed) return buffer

  let size = buffer === null ? 64 : buffer.length
  while (size < needed) size *= 2
  return new Int32Array(size)
}

/** Grow a scratch buffer without invalidating offsets into its existing data. */
function grownPreserving(buffer: Int32Array, needed: number): Int32Array {
  if (buffer.length >= needed) return buffer

  let size = buffer.length
  while (size < needed) size *= 2
  const next = new Int32Array(size)
  next.set(buffer)
  return next
}

/**
 * Set `buffer[start..end)` to `value`.
 *
 * `TypedArray.prototype.fill` is a builtin call whose setup does not depend on
 * how much it then writes, so for the handful of words a pattern's mask spans
 * it costs several times what writing them does. These ranges are `words` long
 * — four, for a 128-element pattern — and the mask builder clears one per
 * distinct element, which made `fill` the single largest cost in scoring a
 * medium-length pair. Past roughly 64 words the builtin's wider stores win and
 * it takes over again.
 */
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

/**
 * Grow the shared table until `symbol` indexes it, and hand back both halves.
 *
 * Called from the mask builders only, and only on the first element that does
 * not fit — at most eight times in the life of the process, so returning a pair
 * rather than reading the two module variables back costs nothing measurable
 * and keeps the caller from having to re-narrow them.
 *
 * The old contents carry over. A builder that widens partway through a pattern
 * has already filed earlier elements under the narrower limit, and every one of
 * those was below it, so where they sit does not change. The caller passes the
 * halves it is already holding, which is both the copy source and the proof
 * that there is one.
 */
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

/**
 * How far the generation counter runs before the stamps are cleared and it
 * starts again.
 *
 * A stamp is an `Int32Array` cell, so the counter cannot exceed what one holds.
 * Reaching the ceiling takes two billion mask builds in a single process, which
 * no test is going to sit through — {@link resetBitVectorScratch} takes a
 * starting generation so that the wrap can be driven directly instead.
 */
const GENERATION_LIMIT = 0x7fff_ffff

/** Stamps start at 0 and generations at 1, so a stale slot can never match. */
function nextGeneration(): number {
  generation++
  if (generation >= GENERATION_LIMIT) {
    directStamps().fill(0)
    generation = 1
  }

  return generation
}

/**
 * Drop the overflow map's contents, skipping the call when it holds nothing.
 *
 * Every mask builder has to clear this map: a symbol above `directLimit` is
 * answered out of it, and a stale entry left by an earlier call would report a
 * match against a pattern that does not contain the symbol. The table beside it
 * needs no such pass because {@link nextGeneration} stamps it instead.
 *
 * The map is empty for every input that stays inside the direct table, which is
 * all Latin-1 text and — once the table has widened — most other scripts too.
 * `Map.prototype.clear` does not care: it is a builtin call whose cost is in
 * being called rather than in what it erases, and it was measured at 28 ns per
 * comparison on eight-character pairs and 50 ns on thirty-two-character ones,
 * against a total of 53 ns and 302 ns for the same work without it. Reading
 * `size` first is the same clear, minus the call nothing needed.
 */
function clearWide(wide: Map<unknown, number>): void {
  if (wide.size !== 0) wide.clear()
}

/**
 * Match masks for a single-word pattern: each distinct element maps to a
 * bitmask of the positions it occupies.
 */
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

/**
 * Match masks for a multi-word pattern. The ASCII table and the `Map` hold an
 * offset into `maskPool` rather than a mask, since a mask is now `words` wide.
 * A negative offset means "this element does not occur in the pattern".
 */
export function buildBlockMasks(
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

  let pool = maskPoolOf()
  let distinct = 0
  let limit = directLimit
  const stringPattern = typeof pattern === 'string'

  // A normalized string contains only integer UTF-16 code units. Keeping that
  // monomorphic case out of the generic loop removes the type, integer, NaN and
  // Map decisions from every pattern element; only widening the direct table can
  // still be necessary. Multi-word string comparisons dominate the LCS scorers.
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

    // As in `buildWordMasks`: only an element already known to belong in the
    // table asks whether the table is wide enough for it yet.
    if (direct && symbol >= limit) {
      const widened = widenDirect(symbol, slots, stamps)
      slots = widened[0]
      stamps = widened[1]
      limit = directLimit
    }

    // An unmatchable element gets no slot at all, so it can never light up a
    // bit for any position — including its own.
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

/**
 * Which pattern the shared mask table currently holds, so scoring one query
 * against a list of candidates builds its masks once rather than once per
 * candidate — upstream's `CachedLevenshtein`, without a second copy of the
 * kernels to read a `PatternMask` instead of the table.
 *
 * `maskGeneration` is what makes reuse safe: any other mask build bumps the
 * generation, and a stale table then fails the test rather than being read.
 *
 * Only strings are remembered. Every other sequence is mutable, and a caller
 * that changes one between two calls would otherwise be scored against the
 * masks of what it used to hold. A string cannot change under us.
 *
 * Remembering one holds it live until another replaces it, so a single call
 * over a huge string would keep that string reachable for as long as nothing
 * else was scored. {@link MASK_PATTERN_LIMIT} is what stops that, and it costs
 * nothing to draw: the build is one pass over the pattern while the comparison
 * that follows is that pass times a word per 32 elements, so the share the
 * cache saves shrinks as the pattern grows. Measured, reusing the masks is
 * worth 1.9x at 64 elements, 1.2x at 256, and inside the noise from 1024 up.
 */
const MASK_PATTERN_LIMIT = 4096
let maskPattern: string | null = null
let maskStart = -1
let maskLength = -1
let maskWords = -1
let maskGeneration = 0

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

/** Longest input the single-word kernels can take as their pattern. */
export const WORD_LIMIT: number = WORD_BITS

/**
 * A miss budget meaning "no bound" — larger than any indel distance two
 * sequences addressable in JavaScript could have.
 */
export const UNBOUNDED_MISSES: number = Number.MAX_SAFE_INTEGER

// Where two sequences stop agreeing at each end. Written to module state rather
// than returned in an object, so scoring a pair allocates nothing at all.
export let affixPrefix = 0
export let affixLen1 = 0
export let affixLen2 = 0

/**
 * Measure the common prefix and suffix of two ranges, leaving the remaining
 * middles in `affixLen1` / `affixLen2` and the prefix length in `affixPrefix`.
 *
 * Trimming is not only an optimisation: it narrows the bit vector, which is
 * what keeps near-identical inputs — the realistic case — inside the
 * single-word kernel.
 */
export function measureAffix(
  s1: ArrayLike<unknown>,
  start1: number,
  len1: number,
  s2: ArrayLike<unknown>,
  start2: number,
  len2: number,
): void {
  // Indexing a string yields a one-character string, so `s1[i] === s2[i]` below
  // is a string comparison — it interns a character per element and compares by
  // value. `charCodeAt` compares two integers for the same answer, since a pair
  // that reached the kernels as strings is BMP-only and equal code units mean
  // equal characters. Worth splitting: this runs on both ends of every pair and
  // walks the whole shared prefix, which for the near-identical inputs the
  // trimming exists to serve is nearly the whole input.
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

/** {@link measureAffix} over code units. See the comment there. */
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

/**
 * The row vectors, reached through accessors because a kernel in another module
 * cannot assign an imported binding. `grown` discards the old contents, which
 * is safe here: every one of these is refilled before it is read.
 */
export function rowVector(words: number): Int32Array {
  vectorP = grown(vectorP, words)
  return vectorP
}

/** The second row vector — see {@link rowVector}. */
export function rowVectorN(words: number): Int32Array {
  vectorN = grown(vectorN, words)
  return vectorN
}

/** The banded kernels' score row — see {@link rowVector}. */
export function bandVector(words: number): Int32Array {
  bandScores = grown(bandScores, words)
  return bandScores
}

/**
 * Drop every shared buffer and reset the table to its initial width.
 *
 * Correctness does not depend on it: every buffer is refilled before it is
 * read, so dropping one only costs the next call an allocation. Nothing in
 * `src` calls it — the benchmark harness does, so that a case which runs after
 * a 16,384-element pair does not measure faster for the allocation that pair
 * already paid.
 *
 * `startGeneration` is the one thing here that is not about allocation. The
 * counter is otherwise only reachable one build at a time, and the wrap at
 * {@link GENERATION_LIMIT} is two billion builds away; starting near it is what
 * lets that wrap be driven and its stale-stamp clearing checked.
 */
export function resetBitVectorScratch(startGeneration = 0): void {
  maskPool = null
  vectorP = null
  vectorN = null
  asciiSlot = null
  asciiStamp = null
  wideSlot = null
  bandScores = null
  directLimit = DIRECT_LOOKUP_LIMIT
  generation = startGeneration

  maskPattern = null
  maskStart = -1
  maskLength = -1
  maskWords = -1
  maskGeneration = 0
}
