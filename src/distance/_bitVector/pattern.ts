/**
 * Immutable match masks for one prepared pattern.
 *
 * Distinct from the shared table in `shared.ts` in the one way that matters:
 * this storage belongs to the pattern that built it. A later kernel call cannot
 * overwrite it, so a prepared process scorer can hold one for an entire
 * extraction or matrix row — which is what `partialRatio`, `cdist` and the
 * prepared scorers in `fuzz.ts` rely on.
 *
 * Nothing here reads or writes the shared scratch, so this module imports no
 * sibling.
 *
 * ## One array, three regions
 *
 * {@link PatternMask.masks} holds every mask, and {@link patternBase} turns any
 * element into one index into it:
 *
 * - `[0, highStart)` — Latin-1, at `symbol * words`, as it always was.
 * - `[highStart, strayStart)` — a window over the pattern's *own* high symbols,
 *   at `highStart + (symbol - highBase) * words`.
 * - the rest — a slot per element that neither region can address, reached
 *   through {@link PatternMask.wideOffsets}.
 *
 * The window is what makes non-Latin text cheap. Text outside Latin-1 used to
 * take a `Map.get` per element of every candidate, which measured 1.62x to
 * 1.91x of the whole prepared comparison once the recurrence itself got fast —
 * Cyrillic 1.84x, Greek 1.78x, Hebrew 1.91x, CJK 1.62x. A window costs a
 * subtract and a bounds test instead, and costs nothing in memory: it replaces
 * the `Map` rather than joining it, and a single-script pattern spans a few
 * dozen code points, not the whole block. A 40-element Cyrillic pattern is 576
 * ints windowed against 564 ints plus a 26-entry `Map` before.
 *
 * A pattern mixing distant scripts is not windowed at all, and every high
 * element takes a stray slot instead. Two bounds decide that, and the second is
 * the load-bearing one: {@link WINDOW_SPAN_LIMIT} on the code points spanned,
 * and {@link WINDOW_CELL_LIMIT} on `span * words` — the cells actually
 * allocated, which the span alone does not bound, since the same span behind a
 * 4096-element pattern is thirty times the storage it is behind a short one.
 * Together they cap a window at 64KB whatever the pattern's length, which
 * matters because `cdist` holds one prepared pattern per row.
 *
 * Addressing all three regions the same way is what makes the lookup a single
 * number. While the masks lived in two arrays it had two things to say — which
 * array and which offset — and encoding both in one number cost 0.79-0.84x on
 * Cyrillic, because the decode landed on every element of the path the wide
 * table exists to make fast.
 *
 * That does not make it a shared function: see {@link patternBase} for why each
 * kernel carries its own copy of the body instead.
 */

// Declared here rather than imported from `shared.ts`. These are read once per
// element of the pattern, and a cross-module binding does not fold the way a
// module-local `const` does — measured at +3% on Latin-1 and +15% on Cyrillic
// for a loop of this shape. `_bitVector/shared.ts` is the canonical home and
// documents the invariant these have to keep; the values must agree with it.
const WORD_SHIFT = 5
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

/**
 * Widest span of high elements a pattern will window, in code points.
 *
 * A pattern mixing Cyrillic with CJK spans tens of thousands of code points to
 * hold a few dozen elements; past this it takes stray slots instead.
 */
const WINDOW_SPAN_LIMIT = 2048

/**
 * Widest window a pattern will allocate, in mask cells.
 *
 * The span alone does not bound the memory — a window is `span * words` cells,
 * so the same 2048-code-point span is 32KB behind a four-word pattern and 1MB
 * behind a 4096-element one. This is the bound that actually holds: 16384 cells
 * is 64KB whatever the pattern's length, which matters because `cdist` holds
 * one prepared pattern per row.
 *
 * A cap rather than a measured optimum. It is set where the allocation stops
 * being obviously worth it, not where scoring stops improving.
 */
const WINDOW_CELL_LIMIT = 16_384

/** Stand-in for the stray map a pattern inside the window never fills. */
const EMPTY_WIDE: ReadonlyMap<unknown, number> = new Map<unknown, number>()

/**
 * The masks of an empty pattern, which every kernel returns before reading.
 *
 * Shared rather than built, so `preparePattern('')` allocates nothing — without
 * it the Latin-1 block alone was a kilobyte of storage no one can read, and
 * `words` being zero left the builder carrying a stride of one that meant
 * nothing.
 */
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

/** Immutable match masks for one reusable pattern. */
export interface PatternMask {
  readonly length: number
  readonly words: number
  /** Every mask, in the three regions described above. */
  readonly masks: Int32Array
  /** Lowest high element the window covers. */
  readonly highBase: number
  /** How many elements the window covers; `0` when there is no window. */
  readonly highCount: number
  /** Where the window starts in {@link masks}. */
  readonly highStart: number
  /** Absolute index into {@link masks} for an element neither region holds. */
  readonly wideOffsets: ReadonlyMap<unknown, number>
}

/** Whether `symbol` indexes the Latin-1 region directly. */
function isDirectSymbol(symbol: unknown): boolean {
  return (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  )
}

/** Whether `symbol` is an integer the window could cover. */
function isHighSymbol(symbol: unknown): symbol is number {
  return (
    typeof symbol === 'number' && symbol >= DIRECT_LOOKUP_LIMIT && (symbol | 0) === symbol
  )
}

/**
 * Where this element's masks start in {@link PatternMask.masks}, or `-1` if the
 * pattern does not hold it.
 *
 * `NaN` is spelled `symbol === symbol` and given no slot, so it matches nothing
 * — every other comparison in this library is `===`, under which `NaN` does not
 * equal itself, and a `Map` would say it does.
 *
 * ## This is the definition; the kernels carry copies
 *
 * {@link preparePattern} below is the only caller, and it runs once per pattern.
 * Every kernel that reads a `PatternMask` writes this body out at its own call
 * site instead, and that is not a style choice.
 *
 * Shared, it is called with whatever any kernel is scoring: numbers when the
 * input is a string, one-character strings and arbitrary objects when it is an
 * array. One function seeing all of them goes megamorphic and stops
 * specialising — measured at **2.43x slower** once other call sites had used it,
 * against a module-local copy that did not move. Across the suite that showed up
 * as `partialRatio` at 0.49x to 0.59x and the whole thing at -13.5%, which is
 * how it was found: the micro-benchmark that said the call was free exercised
 * one site with one element type, which is exactly the case that cannot see
 * this.
 *
 * So each copy specialises on the elements its own kernel sees. `shared.ts`
 * documents the same rule for the analogous test against the shared table; this
 * is that rule again, for the same reason, at a larger cost. Any copy that
 * disagrees with this body is a bug.
 */
export function patternBase(
  symbol: unknown,
  words: number,
  highBase: number,
  highCount: number,
  highStart: number,
  wideOffsets: ReadonlyMap<unknown, number>,
): number {
  if (
    typeof symbol === 'number' &&
    symbol >= 0 &&
    symbol < DIRECT_LOOKUP_LIMIT &&
    (symbol | 0) === symbol
  ) {
    return symbol * words
  }
  if (typeof symbol === 'number' && (symbol | 0) === symbol) {
    const shifted = symbol - highBase
    if (shifted >= 0 && shifted < highCount) return highStart + shifted * words
  }
  if (symbol === symbol) {
    const offset = wideOffsets.get(symbol)
    if (offset !== undefined) return offset
  }
  return -1
}

/**
 * Build the match masks for one pattern and keep them for repeated scoring.
 *
 * `partialRatio` scores a single pattern against O(n) windows of the other
 * input; left to a range kernel each window rebuilds the very same table, which
 * for a multi-word pattern costs more than the scoring does.
 *
 * The returned storage belongs only to this pattern. Later kernel calls cannot
 * overwrite it, so prepared process scorers can safely keep it for an entire
 * extraction or matrix row. Each preparation still builds from the current
 * sequence contents, so caller mutations cannot surface stale global state.
 *
 * One pass over the pattern whenever it holds nothing above Latin-1, which is
 * the common case and the one `levenshteinSmallBand` builds for a *single*
 * comparison. That case is the reason the loop is shaped this way: the window's
 * size is not known until the pattern has been read, and sizing it with a
 * separate first walk cost an all-Latin one-shot preparation 0.84x.
 *
 * So the Latin-1 bits are written as the pattern is read, and anything else is
 * remembered by position. A pattern that has such elements then grows the table
 * once — the block already written moves rather than being rebuilt — and walks
 * only those positions, twice at worst, when the span turns out too wide to
 * window. Never the whole pattern again.
 */
export function preparePattern(
  pattern: ArrayLike<unknown>,
  start: number,
  length: number,
  step = 1,
): PatternMask {
  if (length === 0) return EMPTY_PATTERN

  const words = (length + WORD_MASK) >>> WORD_SHIFT
  const stringPattern = typeof pattern === 'string'
  const directCells = DIRECT_LOOKUP_LIMIT * words

  // One pass, writing the Latin-1 bits as it goes. A pattern with nothing above
  // Latin-1 — the overwhelmingly common one, and the one `levenshteinSmallBand`
  // builds for a single comparison — is finished when this loop ends: one walk,
  // one allocation, no copy. Walking twice to size the table first cost it
  // 0.84x, which is the whole reason this is shaped the way it is.
  //
  // Everything else is remembered by position in `deferred`, so the work below
  // walks those elements rather than the pattern again.
  let masks = new Int32Array(directCells)
  let low = 0
  let high = -1
  let deferred: number[] | null = null
  let strays: Map<unknown, number> | null = null

  for (let i = 0; i < length; i++) {
    const index = start + i * step
    const symbol = stringPattern ? pattern.charCodeAt(index) : pattern[index]
    if (isDirectSymbol(symbol) && typeof symbol === 'number') {
      masks[symbol * words + (i >>> WORD_SHIFT)] |= 1 << (i & WORD_MASK)
      continue
    }
    // `NaN` matches nothing, so it is given no slot at all.
    if (symbol !== symbol) continue
    if (deferred === null) deferred = []
    deferred.push(i)
    if (isHighSymbol(symbol)) {
      if (high < 0 || symbol < low) low = symbol
      if (symbol > high) high = symbol
      continue
    }
    if (strays === null) strays = new Map<unknown, number>()
    strays.set(symbol, 0)
  }

  if (deferred === null) {
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

  // Too wide to window: those elements need slots of their own after all. Only
  // the deferred positions are revisited, never the whole pattern.
  if (!windowed) {
    for (let d = 0; d < deferred.length; d++) {
      const index = start + deferred[d] * step
      const symbol = stringPattern ? pattern.charCodeAt(index) : pattern[index]
      if (!isHighSymbol(symbol)) continue
      if (strays === null) strays = new Map<unknown, number>()
      strays.set(symbol, 0)
    }
  }

  const highStart = directCells
  const strayStart = highStart + highCount * words
  let strayCount = 0
  if (strays !== null) {
    for (const key of strays.keys()) {
      strays.set(key, strayStart + strayCount * words)
      strayCount++
    }
  }

  // The Latin-1 block is already filled, so it moves rather than being rebuilt.
  const grown = new Int32Array(strayStart + strayCount * words)
  grown.set(masks)
  masks = grown

  const wideOffsets = strays ?? EMPTY_WIDE
  for (let d = 0; d < deferred.length; d++) {
    const i = deferred[d]
    const index = start + i * step
    const symbol = stringPattern ? pattern.charCodeAt(index) : pattern[index]
    const base = patternBase(symbol, words, highBase, highCount, highStart, wideOffsets)
    if (base < 0) continue
    masks[base + (i >>> WORD_SHIFT)] |= 1 << (i & WORD_MASK)
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
