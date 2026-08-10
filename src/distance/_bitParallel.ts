/**
 * Bit-parallel alignment matrices (Hyyrö 2004), used to recover edit
 * operations. Ports of the `_matrix` helpers in `LCSseq_py.py` and
 * `Levenshtein_py.py`.
 *
 * The vectors are as wide as `s1`, so they are held across machine words with
 * the carries chained by hand. Upstream leans on Python's arbitrary-precision
 * integers here, and `BigInt` would transfer that line for line — but a word
 * matrix recovers the same alignment between 1.5x and 3.3x faster, so the
 * arithmetic is spelled out instead.
 *
 * Each metric then has two kernels rather than one. A pattern of at most 32
 * elements — which, after affix trimming, is most of what fuzzy matching
 * compares — holds its whole vector in a number, so the one-word kernels carry
 * nothing between words, store a row by assignment, and never allocate the
 * per-word machinery the general case needs.
 *
 * @internal Inputs must already be normalized by `conv` / `convSequence`.
 * In particular, raw strings would otherwise be indexed as UTF-16 code units.
 */

function popcount32(word: number): number {
  let bits = word - ((word >>> 1) & 0x5555_5555)
  bits = (bits & 0x3333_3333) + ((bits >>> 2) & 0x3333_3333)
  return (((bits + (bits >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24
}

/**
 * How much wider than the pattern itself a direct table may be.
 *
 * The table has an entry per element value between the pattern's least and its
 * greatest, so text drawn from one alphabet fills most of it and text drawn
 * from two distant ones — Latin plus CJK — would leave almost all of it empty.
 * Past this the pattern takes the `Map`, which costs a lookup per row but not a
 * table far larger than the pattern that has to be zeroed to build it.
 *
 * The slack keeps short patterns of ordinary text on the table: two characters
 * of Latin-1 span at most 256 entries however far apart in it they sit. Text
 * that spans most of Latin-1 is the case this bound decides, and it decides it
 * in the table's favour on measurement — eight such characters pay about 5% for
 * a table they barely fill, and thirty-two of them gain 1.5x.
 */
const SPAN_SLACK = 256

/** Stands in for a table a pattern with no table-able element never fills. */
const NO_MASKS = new Int32Array(0)

/**
 * Range of elements a direct table would have to cover to index a range of
 * `s1` by the element itself.
 *
 * `minSymbol` above `maxSymbol` says there is no such table: nothing satisfies
 * `symbol >= 1 && symbol <= 0`, so every element takes the `Map`. That is the
 * answer for a range holding no indexable element and for one whose elements
 * are too far apart to sit in a table together.
 */
interface SymbolSpan {
  readonly minSymbol: number
  readonly maxSymbol: number
}

const EMPTY_SPAN: SymbolSpan = { minSymbol: 1, maxSymbol: 0 }

function symbolSpan(s1: ArrayLike<unknown>, start: number, length: number): SymbolSpan {
  const stringPattern = typeof s1 === 'string'
  let minSymbol = 1
  let maxSymbol = 0

  for (let i = 0; i < length; i++) {
    const index = start + i
    const symbol = stringPattern ? s1.charCodeAt(index) : s1[index]
    // The integer test is load bearing: a typed array has no element at a
    // fractional index, so `1.5` would be written nowhere and read back as
    // `undefined` rather than as itself. It also excludes NaN and the
    // infinities, none of which is a position in a table.
    //
    // Negative elements need no test of their own — the table is indexed from
    // `minSymbol`, so a sequence of them indexes from its own least element.
    if (typeof symbol !== 'number' || (symbol | 0) !== symbol) continue

    if (minSymbol > maxSymbol) {
      minSymbol = symbol
      maxSymbol = symbol
    } else if (symbol < minSymbol) minSymbol = symbol
    else if (symbol > maxSymbol) maxSymbol = symbol
  }

  return maxSymbol - minSymbol >= length + SPAN_SLACK
    ? EMPTY_SPAN
    : { minSymbol, maxSymbol }
}

/**
 * Match masks for a range that fits one machine word, indexed by the element
 * itself rather than hashed.
 *
 * A pattern of at most 32 elements has at most 32 distinct ones, so the table
 * covers only `[minSymbol, maxSymbol]` — 26 entries for lowercase text —
 * instead of the whole of Latin-1 as the kernels under `_bitVector/` do. Those
 * hold one table for the process and stamp its entries; this one is built and
 * dropped per matrix, where a kilobyte to zero costs more than the lookups it
 * would save.
 *
 * An element outside the span matches nothing, which is the right answer for
 * every element the pattern does not contain — so `wide` stays unbuilt unless
 * the pattern itself holds something the table cannot index.
 */
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

    // The same four-part test the row loops apply, and it has to stay the same:
    // an element filed in one table and looked up in the other reads as absent
    // from a pattern that contains it.
    if (
      typeof symbol === 'number' &&
      symbol >= minSymbol &&
      symbol <= maxSymbol &&
      (symbol | 0) === symbol
    ) {
      direct[symbol - minSymbol] |= bit
      continue
    }

    // `Map` matches keys by SameValueZero, under which NaN equals itself;
    // sequence equality everywhere else is `===`, under which it does not.
    if (symbol !== symbol) continue
    if (wide === null) wide = new Map<unknown, number>()
    wide.set(symbol, (wide.get(symbol) ?? 0) | bit)
  }

  return { direct, minSymbol, maxSymbol, wide }
}

/**
 * The same masks for a range wider than one machine word: each distinct element
 * owns `words` consecutive words of `masks`, and both tables map an element to
 * where its own start rather than to the masks themselves.
 *
 * Block `0` is reserved and stays zeroed, so a base of `0` says the range holds
 * no such element and reads back as matching nothing. That is what lets a table
 * start at zero rather than at `-1`, and what lets the Levenshtein row loop
 * read a word rather than branch on whether there is one.
 */
interface WordMasks extends SymbolSpan {
  readonly masks: Int32Array
  /** Where each element of the span starts in `masks`, or `0` for absent. */
  readonly bases: Int32Array
  readonly wide: ReadonlyMap<unknown, number> | null
}

/**
 * Blocks a mask table starts with when nothing bounds it but the range length.
 *
 * The number of distinct elements is only known once the range has been walked,
 * so either the walk happens twice or the table grows. Growing copies at most
 * twice what it ends up holding, and a copy of machine words is far cheaper
 * than a second pass that has to re-decide which table each element belongs to.
 *
 * A range with no span holds nothing a table can index, so its length is all
 * there is to go on and a long one makes that far too loose a bound to allocate
 * against. The table starts here and doubles.
 */
const INITIAL_BLOCKS = 64

/**
 * Widest span a table is sized against outright, rather than grown into.
 *
 * A span is close to what the range holds only while it is narrow. Once it is
 * wide the two come apart — an element every `n` positions spans as widely as
 * one at every position and fills a table `n` times smaller, and nothing known
 * before the walk tells them apart. So a span reaching past this starts at
 * `INITIAL_BLOCKS` instead; and since the span is exact once reached, a table
 * that outgrows the start goes straight there rather than doubling toward it.
 * A range that does fill a wide span pays one copy of the start for that; one
 * that does not — an array of a few values a thousand apart — never allocates
 * the thousand blocks between them.
 *
 * `SPAN_SLACK` is the same number for the same reason: a table this wide is one
 * ordinary text builds and fills, and past it the range is drawing from
 * somewhere its elements do not densely cover.
 */
const MAX_SPAN_BLOCKS = 256

/**
 * The table grown past `blocks`, which is either exactly `spanSize` or twice
 * over until it fits, capped at `limit`.
 *
 * `spanSize` is where a span puts the last block it can account for, so growth
 * that lands there needs no headroom and takes no second copy. Past it — a range
 * whose elements are not all the span's, so `wide` draws blocks from the same
 * table — and with no span at all, only `limit` bounds the table, and it bounds
 * it by however much the range repeats itself. Doubling is what there is.
 */
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
  // A range holds at most as many distinct elements as it is long, and this is
  // the only bound that holds for both tables at once — the span bounds what the
  // span table takes, and an element it rejects is filed in `wide`, which draws
  // its blocks from the same table.
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

    // The same four-part test the row loops apply, and it has to stay the same:
    // an element filed in one table and looked up in the other reads as absent
    // from a range that contains it.
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
      // `Map` matches keys by SameValueZero, under which NaN equals itself;
      // sequence equality everywhere else is `===`, under which it does not.
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

/** The bits of a word that are positions of a range `length` elements long. */
function validBits(length: number): number {
  return (length & 31) !== 0 ? (1 << (length & 31)) - 1 : -1
}

export interface LcsSeqMatrix {
  readonly sim: number
  /** `s2Length` rows of `words` machine words, row-major. */
  readonly rows: Int32Array
  readonly words: number
}

/**
 * Port of `LCSseq_py._matrix`. Returns the LCS length and the row vectors.
 *
 * Both inputs are given as a range rather than a whole sequence. Recovery
 * always works on an affix-trimmed middle, and materialising that middle cost
 * two arrays and a copy of every element per call — on a divide-and-conquer
 * recovery, per leaf. Reading through an offset costs nothing.
 *
 * @internal Inputs must already be normalized by `conv` / `convSequence`.
 */
export function lcsSeqMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
): LcsSeqMatrix {
  // An empty side has no common subsequence with anything, and recovery reads
  // no row of a matrix with no rows — so neither the masks nor the vector are
  // worth building. `words` is `0` because `rows` holds nothing to index.
  if (s1Length === 0 || s2Length === 0) {
    return { sim: 0, rows: new Int32Array(0), words: 0 }
  }

  const words = (s1Length + 31) >>> 5
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
  // Once a state row no longer fits comfortably in the fast caches, keeping a
  // small scratch vector hot and copying it out beats reading the preceding
  // matrix row back in. The crossover was measured at 1024 pattern elements:
  // in-place rows win through 32 words, while an 8192-element matrix regresses.
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

  // Keep the initial all-ones vector directly before the returned rows. Each
  // row can then read its predecessor and write its final position in one pass,
  // instead of writing a scratch vector and copying that vector afterwards.
  // The allocation is the same size as the former rows + scratch allocation;
  // the returned view simply hides the initial vector from recovery.
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
        // `S + u` modulo the word width, with the carry out recovered from the
        // operands: `S - u` is `S & ~u` here, since `u` is a subset of `S`.
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        storage[rowBase + w] = sum | (s & ~u)
      }
    } else {
      storage.copyWithin(rowBase, previousBase, rowBase)
    }
  }

  // Only the bits below `s1Length` are positions; the rest of the top word was
  // never a cell and must not be counted.
  let sim = 0
  for (let w = 0; w < words; w++) {
    const valid = w === words - 1 ? validBits(s1Length) : -1
    sim += popcount32(~storage[s2Length * words + w] & valid)
  }

  return { sim, rows, words }
}

/**
 * `lcsSeqMatrix` for a pattern of at most 32 elements: the vector is a number,
 * so the row loop carries nothing between words and stores a row by assignment.
 */
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

/**
 * Port of `Levenshtein_py._matrix`. Returns the distance and the row vectors.
 *
 * @internal Inputs must already be normalized by `conv` / `convSequence`.
 */
export interface LevenshteinMatrix {
  readonly dist: number
  /** `s2Length` rows of `stride` words, row-major. */
  readonly vp: Int32Array
  readonly vn: Int32Array
  readonly stride: number
  /**
   * Bit position that word `0` of each row starts at, or `null` when every row
   * holds the whole vector. Always a multiple of the word width.
   */
  readonly offsets: Int32Array | null
}

/**
 * Whether a matrix over these lengths stores a band rather than whole rows.
 *
 * A band has to be narrower than the vector to be worth storing, and a vector
 * of one word has no band to narrow — the one-word kernel below ignores the
 * caller's distance for exactly that reason.
 */
function bandedRows(s1Length: number, maximumDistance: number): boolean {
  return s1Length > 32 && maximumDistance >= 0 && 2 * maximumDistance + 1 < s1Length
}

/**
 * Words each stored row takes.
 *
 * A band is `2 * maximumDistance + 1` positions wide and is stored from the
 * word boundary at or below its first one, so at worst it starts 31 bits into
 * the first stored word: `ceil((width + 31) / 32)` words hold any placement.
 */
function rowStride(s1Length: number, maximumDistance: number): number {
  const words = (s1Length + 31) >>> 5
  return bandedRows(s1Length, maximumDistance)
    ? Math.min(words, (2 * maximumDistance + 63) >>> 5)
    : words
}

/**
 * Bytes {@link levenshteinMatrix} will allocate for its rows.
 *
 * Exported so a caller deciding whether it can afford one asks the code that
 * does the allocating. The estimate it replaced was written for row objects
 * holding a payload each, and outlived them: it charged 24 bytes of overhead
 * per row plus the band's bits, where the rows are now two flat `Int32Array`s
 * and an offset. On a narrow band it read two to three times high, which sent
 * alignments to the divide-and-conquer path — and a different, though equally
 * short, edit script — over matrices that would have fitted comfortably.
 */
export function levenshteinMatrixBytes(
  s1Length: number,
  s2Length: number,
  maximumDistance: number,
): number {
  const banded = bandedRows(s1Length, maximumDistance)
  return s2Length * (2 * rowStride(s1Length, maximumDistance) * 4 + (banded ? 4 : 0))
}

/**
 * Levenshtein matrix, optionally keeping only the caller-proven edit band of
 * each row. The recurrence is exact either way; a band narrows what is stored.
 */
export function levenshteinMatrix(
  s1: ArrayLike<unknown>,
  s1Start: number,
  s1Length: number,
  s2: ArrayLike<unknown>,
  s2Start: number,
  s2Length: number,
  maximumDistance = -1,
): LevenshteinMatrix {
  // With one side empty every element of the other is an edit, and recovery
  // reads no row of a matrix with no rows.
  if (s1Length === 0 || s2Length === 0) {
    return {
      dist: s1Length + s2Length,
      vp: new Int32Array(0),
      vn: new Int32Array(0),
      stride: 0,
      offsets: null,
    }
  }

  const words = (s1Length + 31) >>> 5
  // A band cannot narrow a row that is already one word wide, so the one-word
  // kernel ignores `maximumDistance` and stores every row whole.
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
      // A `matchBase` of `0` is the reserved zero block, so an element `s1` does
      // not hold reads its masks like any other rather than testing for them.
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

    // The band of row `j + 1` starts here; storing from the word boundary at or
    // below it keeps the recovery's index arithmetic to a shift.
    const from = Math.min(words - stride, Math.max(0, j + 1 - maximumDistance - 1) >>> 5)
    offsets[j] = from << 5
    for (let w = 0; w < stride; w++) {
      vp[base + w] = vpState[from + w]
      vn[base + w] = vnState[from + w]
    }
  }

  return { dist: currDist, vp, vn, stride, offsets }
}

/**
 * `levenshteinMatrix` for a pattern of at most 32 elements: both vectors are
 * numbers, so a row is one Myers step with the carries folded into constants.
 */
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

/** Length of the shared prefix and suffix, the suffix measured after the prefix. */
export function commonAffix(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
): { prefixLen: number; suffixLen: number } {
  const shorter = Math.min(s1.length, s2.length)
  const end1 = s1.length - 1
  const end2 = s2.length - 1

  let prefixLen = 0
  let suffixLen = 0

  // Comparing two strings position by position allocates a one-character string
  // per side per step; reading the code units compares integers instead. Both
  // inputs share a representation by the time they arrive, so the branch is
  // taken once rather than per position.
  if (typeof s1 === 'string' && typeof s2 === 'string') {
    while (prefixLen < shorter && s1.charCodeAt(prefixLen) === s2.charCodeAt(prefixLen)) {
      prefixLen++
    }
    while (
      suffixLen < shorter - prefixLen &&
      s1.charCodeAt(end1 - suffixLen) === s2.charCodeAt(end2 - suffixLen)
    ) {
      suffixLen++
    }

    return { prefixLen, suffixLen }
  }

  while (prefixLen < shorter && s1[prefixLen] === s2[prefixLen]) prefixLen++
  while (
    suffixLen < shorter - prefixLen &&
    s1[end1 - suffixLen] === s2[end2 - suffixLen]
  ) {
    suffixLen++
  }

  return { prefixLen, suffixLen }
}

/** @internal True when position `pos` of row `row` of a word matrix is set. */
export function rowBitSet(
  rows: Int32Array,
  words: number,
  row: number,
  pos: number,
): boolean {
  return (rows[row * words + (pos >>> 5)] & (1 << (pos & 31))) !== 0
}

/** @internal Test a whole-vector position in a row stored from `offset`. */
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
