/**
 * Hyyrö's bit-parallel LCS, in the "revisited" form upstream uses.
 *
 * The whole family lives in one module — the single-word and multi-word
 * kernels, the banded variant, the mbleven models and the prepared-mask
 * readers — because {@link lcsLengthRange} is only legible beside the kernels
 * it chooses between.
 *
 * This is where nearly all of the library's time goes: `fuzz.ratio`, `WRatio`
 * and therefore `process.extract` all bottom out here.
 */

import type { PatternMask } from './pattern.js'
import {
  affixLen1,
  affixLen2,
  affixPrefix,
  directSlots,
  directStamps,
  blockMasksFor,
  buildWordMasks,
  clearRange,
  directLimit,
  maskPoolOf,
  measureAffix,
  rowVector,
  UNBOUNDED_MISSES,
  wideSlots,
} from './shared.js'

// Declared here rather than imported — see the note in `shared.ts`, which holds
// the canonical definitions. Read once per element, where a cross-module
// binding does not fold the way a module-local `const` does.
const WORD_BITS = 32
const WORD_SHIFT = 5
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

/** Duplicated for the same reason as the constants: called once per word. */
function popcount(word: number): number {
  let bits = word - ((word >>> 1) & 0x5555_5555)
  bits = (bits & 0x3333_3333) + ((bits >>> 2) & 0x3333_3333)
  return (((bits + (bits >>> 4)) & 0x0f0f_0f0f) * 0x0101_0101) >>> 24
}

/**
 * Hyyrö's bit-parallel LCS over one word, in the "revisited" form upstream
 * uses.
 *
 * `s` starts all-ones and a bit is cleared at each pattern position where the
 * LCS grows, so the answer is `popcount(~s)`. Bits above `patternLength` can
 * never be cleared — the pattern has no match there, so `u` is zero and
 * `s & ~u` holds them — which is why the word needs no masking.
 *
 * The mask lookup is written out rather than called: it runs once per element
 * of the longer input, and a converted string is entirely numbers, so the
 * branch predicts perfectly and the loop stays free of call overhead.
 */
function lcsOneWord(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  return lcsOneWordStamped(
    buildWordMasks(pattern, patternStart, patternLength),
    text,
    textStart,
    textLength,
  )
}

/** {@link lcsOneWord} against masks that are already built. */
function lcsOneWordStamped(
  stamp: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  let s = -1
  const limit = directLimit

  // The text's own loop when it is a string. `charCodeAt` yields an integer in
  // `0..0xFFFF` and nothing else, so the range and integrality tests below are
  // answered in advance and the `NaN` arm has nothing to catch — one compare
  // against `limit` is all that is left.
  //
  // This is the loop `ratio` and `wRatio` spend most of their time in, which is
  // what earns it the duplication — see the note on the generic arm below.
  if (typeof text === 'string') {
    for (let i = 0; i < textLength; i++) {
      const symbol = text.charCodeAt(textStart + i)
      // No overflow-map arm, matching {@link lcsFourWordsStamped}'s string
      // branch. `buildWordMasks` files a symbol in that map only when it is not
      // an integer in `0..0xFFFF` — everything else widens the direct table
      // instead — and `charCodeAt` produces nothing else. So a code unit at or
      // above `limit` is one the pattern cannot contain, and contributes no
      // matches. Audited by throwing on a hit here: the suite passed, and so did
      // a sweep of 47,520 string calls over every mask region, as pattern and
      // as text, at every one-word width.
      const matches = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : 0

      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
    }

    return popcount(~s)
  }

  for (let i = 0; i < textLength; i++) {
    const symbol = text[textStart + i]
    let matches: number

    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      matches = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < limit &&
      (symbol | 0) === symbol
    ) {
      matches = stamps[symbol] === stamp ? slots[symbol] : 0
    } else if (symbol === symbol) {
      matches = wide.get(symbol) ?? 0
    } else {
      matches = 0
    }

    // `u` is a subset of `s` by construction, so `s - u` is `s & ~u` — no
    // borrow can leave the word, and only the addition needs a carry chain.
    const u = s & matches
    s = (s + u) | 0 | (s & ~u)
  }

  return popcount(~s)
}

/** The same recurrence carried across as many words as the pattern needs. */
function lcsManyWords(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const words = (patternLength + WORD_BITS - 1) >>> WORD_SHIFT
  const stamp = blockMasksFor(pattern, patternStart, patternLength, words)

  if (words === 4) return lcsFourWordsStamped(stamp, text, textStart, textLength)
  return lcsManyWordsStamped(stamp, words, text, textStart, textLength)
}

/** Four words of LCS state held in locals, for patterns of 97 to 128 elements. */
function lcsFourWordsStamped(
  stamp: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  let s0 = -1
  let s1 = -1
  let s2 = -1
  let s3 = -1
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    let offset: number

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else {
        offset = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : -1
      }
    } else {
      const symbol = text[textStart + i]
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (
        typeof symbol === 'number' &&
        symbol >= DIRECT_LOOKUP_LIMIT &&
        symbol < limit &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (symbol === symbol) {
        offset = wide.get(symbol) ?? -1
      } else {
        offset = -1
      }
    }

    if (offset < 0) continue

    let u = s0 & pool[offset]
    let sum = (s0 + u) | 0
    let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & pool[offset + 1]
    sum = (s1 + u + carry) | 0
    carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
    s1 = sum | (s1 & ~u)

    u = s2 & pool[offset + 2]
    sum = (s2 + u + carry) | 0
    carry = ((s2 & u) | ((s2 | u) & ~sum)) >>> 31
    s2 = sum | (s2 & ~u)

    u = s3 & pool[offset + 3]
    sum = (s3 + u + carry) | 0
    s3 = sum | (s3 & ~u)
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2) + popcount(~s3)
}

/** {@link lcsManyWords} against masks that are already built. */
function lcsManyWordsStamped(
  stamp: number,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const row = rowVector(words)
  clearRange(row, -1, 0, words)

  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    let offset: number

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else {
        offset = symbol < limit && stamps[symbol] === stamp ? slots[symbol] : -1
      }
    } else {
      const symbol = text[textStart + i]
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (
        typeof symbol === 'number' &&
        symbol >= DIRECT_LOOKUP_LIMIT &&
        symbol < limit &&
        (symbol | 0) === symbol
      ) {
        offset = stamps[symbol] === stamp ? slots[symbol] : -1
      } else if (symbol === symbol) {
        offset = wide.get(symbol) ?? -1
      } else {
        offset = -1
      }
    }

    // An element the pattern does not contain leaves the row exactly as it was:
    // with `u` zero the sum is `s`, the carry out is `s & ~s` and so stays zero,
    // and the write back is `s | s`. Every word therefore reproduces itself, and
    // the whole row can be skipped rather than rewritten. Testing it here also
    // takes the branch out of the word loop, where it ran once per word.
    if (offset < 0) continue

    let carry = 0
    let w = 0

    // Upstream advances a few adjacent blocks per loop iteration as well. Eight
    // 32-bit words keep the dependent carry chain explicit while removing most
    // of the loop-counter and bounds-check work from long patterns.
    const unrolledEnd = words & ~7
    for (; w < unrolledEnd; w += 8) {
      let s = row[w]
      let u = s & pool[offset + w]
      let sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w] = sum | (s & ~u)

      s = row[w + 1]
      u = s & pool[offset + w + 1]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 1] = sum | (s & ~u)

      s = row[w + 2]
      u = s & pool[offset + w + 2]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 2] = sum | (s & ~u)

      s = row[w + 3]
      u = s & pool[offset + w + 3]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 3] = sum | (s & ~u)

      s = row[w + 4]
      u = s & pool[offset + w + 4]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 4] = sum | (s & ~u)

      s = row[w + 5]
      u = s & pool[offset + w + 5]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 5] = sum | (s & ~u)

      s = row[w + 6]
      u = s & pool[offset + w + 6]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 6] = sum | (s & ~u)

      s = row[w + 7]
      u = s & pool[offset + w + 7]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w + 7] = sum | (s & ~u)
    }

    for (; w < words; w++) {
      const s = row[w]
      const u = s & pool[offset + w]

      // `(s + u + carry) | 0` is the addition modulo 2^32, and the carry out of
      // the top bit is then recovered from the operands and the truncated sum.
      // Reading it off an unsigned sum instead — `(s >>> 0) + (u >>> 0)` — puts
      // the value in 2^31..2^32, outside the small-integer range, so every
      // iteration of this loop would box a double.
      const sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31

      row[w] = sum | (s & ~u)
    }
  }

  let count = 0
  for (let w = 0; w < words; w++) count += popcount(~row[w])
  return count
}

/** Cutoff-aware Ukkonen word band from upstream's blockwise LCS kernel. */
function lcsManyWordsBanded(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const words = (patternLength + WORD_MASK) >>> WORD_SHIFT
  const stamp = blockMasksFor(pattern, patternStart, patternLength, words)
  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const pool = maskPoolOf()
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const left = patternLength - required
  const right = textLength - required
  let firstWord = 0
  let lastWord = Math.min(words, Math.ceil((left + 1) / WORD_BITS))
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    let offset = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      if (stamps[symbol] === stamp) offset = slots[symbol]
    } else if (
      typeof symbol === 'number' &&
      symbol >= DIRECT_LOOKUP_LIMIT &&
      symbol < limit &&
      (symbol | 0) === symbol
    ) {
      if (stamps[symbol] === stamp) offset = slots[symbol]
    } else if (symbol === symbol) {
      offset = wide.get(symbol) ?? -1
    }

    // An element the pattern does not contain reproduces every word it would
    // touch — the argument in `lcsManyWordsStamped`, which skips the row for it
    // outright. Here the window still has to advance below, so only the
    // recurrence is skipped, not the iteration. It was written as a masked
    // `u = s & (offset < 0 ? 0 : …)` instead, which ran the whole active band
    // to arrive at the row it started with, and also put the test inside the
    // word loop where it ran once per word.
    if (offset >= 0) {
      let carry = 0

      for (let word = firstWord; word < lastWord; word++) {
        const s = row[word]
        const u = s & pool[offset + word]
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[word] = sum | (s & ~u)
      }
    }

    if (i > right) firstWord = Math.floor((i - right) / WORD_BITS)
    // The window for the *next* row, whose highest reachable pattern position
    // is `i + 1 + left` — so the last word is the one that position falls in,
    // and `lastWord` is exclusive. Written as `ceil((i + 1 + left) / 32)` it is
    // one word short whenever that position is the first bit of a word, which
    // drops the match there: two 65-element sequences at `required = 65` lost
    // the match at position 32 and were reported as unreachable.
    if (i + 1 + left <= patternLength) {
      lastWord = Math.min(words, ((i + 1 + left) >>> WORD_SHIFT) + 1)
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  return count >= required ? count : 0
}

/**
 * Edit scripts to try for a given miss budget and length difference — upstream's
 * `lcs_seq_mbleven2018_matrix`.
 *
 * Two bits per operation, least significant first: `01` deletes from the longer
 * side, `10` from the shorter. The row for a budget `k` and length difference
 * `d` lists every script that could reach the optimum, so scanning all of them
 * and keeping the best is exact whenever the true distance is within `k`.
 *
 * Indexed by `(k + k * k) / 2 + d - 1`.
 */
const MBLEVEN_OPS: ReadonlyArray<readonly number[]> = [
  /* k = 1 */
  [], // d = 0 — parity rules this out
  [0x01], // d = 1
  /* k = 2 */
  [0x09, 0x06], // d = 0
  [0x01], // d = 1
  [0x05], // d = 2
  /* k = 3 */
  [0x09, 0x06], // d = 0
  [0x25, 0x19, 0x16], // d = 1
  [0x05], // d = 2
  [0x15], // d = 3
  /* k = 4 */
  [0x96, 0x66, 0x5a, 0x99, 0x69, 0xa5], // d = 0
  [0x25, 0x19, 0x16], // d = 1
  [0x65, 0x56, 0x95, 0x59], // d = 2
  [0x15], // d = 3
  [0x55], // d = 4
]

/** Largest miss budget {@link MBLEVEN_OPS} covers. */
const MBLEVEN_LIMIT = 4

/**
 * LCS length by enumerating the few edit scripts a small miss budget allows.
 *
 * Exact when the true indel distance is at most `budget`; otherwise it returns
 * some length no greater than the true one, which is all the caller needs — a
 * pair that far apart is below its cutoff either way. `longer` must be at least
 * as long as `shorter`.
 */
function lcsMbleven(
  longer: ArrayLike<unknown>,
  longerStart: number,
  longerLength: number,
  shorter: ArrayLike<unknown>,
  shorterStart: number,
  shorterLength: number,
  budget: number,
): number {
  const lengthDiff = longerLength - shorterLength
  const scripts = MBLEVEN_OPS[(budget + budget * budget) / 2 + lengthDiff - 1]
  let best = 0

  for (let s = 0; s < scripts.length; s++) {
    let script = scripts[s]
    let i = 0
    let j = 0
    let common = 0

    while (i < longerLength && j < shorterLength) {
      if (longer[longerStart + i] !== shorter[shorterStart + j]) {
        if (script === 0) break

        if ((script & 1) !== 0) i++
        else j++
        script >>>= 2
      } else {
        common++
        i++
        j++
      }
    }

    if (common > best) best = common
  }

  return best
}

/**
 * Length of the longest common subsequence of two ranges.
 *
 * `budget` bounds the indel distance the caller still cares about: if the two
 * ranges are further apart than that, the result may be any length no greater
 * than the true one. Callers that need the exact value pass
 * {@link UNBOUNDED_MISSES}. Since a caller only ever compares the score it
 * derives against the same cutoff the budget came from, an understated length
 * there and the true length agree on the answer.
 */
export function lcsLengthRange(
  s1: ArrayLike<unknown>,
  start1: number,
  len1: number,
  s2: ArrayLike<unknown>,
  start2: number,
  len2: number,
  budget: number,
): number {
  if (len1 === 0 || len2 === 0) return 0

  // Before either end is walked: a length difference is a lower bound on the
  // indel distance, and trimming a common affix takes the same amount off both
  // sides, so this is the decision the trimmed check below would reach anyway.
  // Reaching it first is what makes a badly sized candidate free to reject,
  // which is most of what an `extract` under a running cutoff does. Answering
  // `0` rather than the affix length is an understatement the contract above
  // allows, and the caller has already rejected the pair either way.
  if (budget < (len1 < len2 ? len2 - len1 : len1 - len2)) return 0

  measureAffix(s1, start1, len1, s2, start2, len2)
  const prefix = affixPrefix
  const middle1 = affixLen1
  const middle2 = affixLen2

  // Every element of a common affix belongs to some optimal LCS, so it can be
  // counted without being compared. Trimming leaves the indel distance — and
  // therefore the budget — untouched.
  const common = len1 - middle1
  if (middle1 === 0 || middle2 === 0) return common

  const lengthDiff = middle1 < middle2 ? middle2 - middle1 : middle1 - middle2

  // No script within the budget can bridge this much length difference, so the
  // caller's cutoff already rejects the pair. A budget of zero is the same
  // story: the middles differ, so at least one miss is unavoidable.
  if (budget < lengthDiff || budget < 1) return common

  // `(budget | 0) === budget` is an integer test only because the comparison
  // before it has already bounded the value: a fractional budget would index
  // past the end of `MBLEVEN_OPS`, and `NaN` or `Infinity` fails the comparison
  // and falls through to the always-exact kernel below. Testing here rather
  // than normalising on entry keeps the check off the path every short input
  // takes.
  if (budget <= MBLEVEN_LIMIT && (budget | 0) === budget) {
    const firstIsLonger = middle1 >= middle2

    return (
      common +
      lcsMbleven(
        firstIsLonger ? s1 : s2,
        (firstIsLonger ? start1 : start2) + prefix,
        firstIsLonger ? middle1 : middle2,
        firstIsLonger ? s2 : s1,
        (firstIsLonger ? start2 : start1) + prefix,
        firstIsLonger ? middle2 : middle1,
        budget,
      )
    )
  }

  // Cost is `ceil(patternLength / 32) * textLength`, so neither side is always
  // the better pattern: below one word the longer side wins because it shortens
  // the streamed side, above it the shorter side wins because it needs fewer
  // words. When both fit in one word the word counts cancel and the rule
  // collapses to "stream the shorter side", which is worth spelling out because
  // it is the case short inputs always take.
  const firstIsPattern =
    middle1 <= WORD_BITS && middle2 <= WORD_BITS
      ? middle2 <= middle1
      : ((middle1 + WORD_MASK) >>> WORD_SHIFT) * middle2 <=
        ((middle2 + WORD_MASK) >>> WORD_SHIFT) * middle1

  const pattern = firstIsPattern ? s1 : s2
  const patternStart = (firstIsPattern ? start1 : start2) + prefix
  const patternLength = firstIsPattern ? middle1 : middle2
  const text = firstIsPattern ? s2 : s1
  const textStart = (firstIsPattern ? start2 : start1) + prefix
  const textLength = firstIsPattern ? middle2 : middle1

  // `common` cancels out of both lengths, so this is `ceil((m1 + m2 - budget) / 2)`
  // over the two middles — and that is at most the shorter of them exactly when
  // their difference is within the budget, which the rejection above settled.
  const requiredTotal = Math.max(0, Math.ceil((len1 + len2 - budget) / 2))
  const requiredMiddle = Math.max(0, requiredTotal - common)
  const fullBand = patternLength + textLength - 2 * requiredMiddle + 1
  const bandWords = Math.min(
    (patternLength + WORD_MASK) >>> WORD_SHIFT,
    Math.floor(fullBand / WORD_BITS) + 2,
  )

  const middle =
    patternLength <= WORD_BITS
      ? lcsOneWord(pattern, patternStart, patternLength, text, textStart, textLength)
      : requiredMiddle > 0 && bandWords < (patternLength + WORD_MASK) >>> WORD_SHIFT
        ? lcsManyWordsBanded(
            pattern,
            patternStart,
            patternLength,
            text,
            textStart,
            textLength,
            requiredMiddle,
          )
        : lcsManyWords(pattern, patternStart, patternLength, text, textStart, textLength)

  return common + middle
}

/**
 * Exact LCS length of the held pattern against `text[textStart, +textLength)`.
 *
 * No common affix is trimmed: trimming would shorten the pattern by a different
 * amount for every text, which is exactly what the held masks cannot express.
 */
export function lcsLengthPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const words = prepared.words
  if (words === 0 || textLength === 0) return 0
  if (words === 4) return lcsFourWordsPrepared(prepared, text, textStart, textLength)

  if (words === 1) {
    let s = -1
    const masks = prepared.masks
    const highBase = prepared.highBase
    const highCount = prepared.highCount
    const highStart = prepared.highStart
    const wideOffsets = prepared.wideOffsets

    // The one-word prepared loop is where `ratio` and every `extract` over
    // short strings spends its time, so the text's representation is settled
    // once, out here, rather than asked of every element. `charCodeAt` yields
    // an integer in `0..0xFFFF`, which answers all four tests the generic arm
    // makes and leaves the `NaN` arm nothing to catch.
    if (typeof text === 'string') {
      for (let i = 0; i < textLength; i++) {
        const symbol = text.charCodeAt(textStart + i)
        let base: number
        if (symbol < DIRECT_LOOKUP_LIMIT) {
          base = symbol
        } else {
          const shifted = symbol - highBase
          base =
            shifted >= 0 && shifted < highCount
              ? highStart + shifted
              : (wideOffsets.get(symbol) ?? -1)
        }
        const matches = base < 0 ? 0 : masks[base]
        const u = s & matches
        s = (s + u) | 0 | (s & ~u)
      }
      return popcount(~s)
    }

    for (let i = 0; i < textLength; i++) {
      const symbol = text[textStart + i]
      // Written out rather than called, and that is load bearing — see the note
      // on `patternBase`, whose body this is. A single shared copy sees numbers
      // from string inputs and strings and objects from array inputs, goes
      // megamorphic, and measured 2.43x slower once other kernels had used it.
      let base = -1
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        base = symbol * 1
      } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 1
            : (wideOffsets.get(symbol) ?? -1)
      } else if (symbol === symbol) {
        base = wideOffsets.get(symbol) ?? -1
      }
      const matches = base < 0 ? 0 : masks[base]
      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
    }
    return popcount(~s)
  }

  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Where the masks start is settled once per element rather than once per
    // word, and an element the pattern does not hold is skipped outright: with
    // no matches every word reproduces itself, so the row comes out of the loop
    // exactly as it went in.
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }
    if (base < 0) continue

    // The two- and three-word widths are written out rather than looped: they
    // cover 33 to 96 elements, which is most of what the fuzz scorers see, and
    // dropping the loop counter and its bounds check is worth 1.15x there.
    // Extracting this into a shared helper gave most of that back — the call
    // does not inline — so it stays spelled out at each kernel that runs it.
    let s = row[0]
    let u = s & masks[base]
    let sum = (s + u) | 0
    let carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[0] = sum | (s & ~u)

    s = row[1]
    u = s & masks[base + 1]
    sum = (s + u + carry) | 0
    carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[1] = sum | (s & ~u)
    if (words === 2) continue

    s = row[2]
    u = s & masks[base + 2]
    sum = (s + u + carry) | 0
    carry = ((s & u) | ((s | u) & ~sum)) >>> 31
    row[2] = sum | (s & ~u)
    if (words === 3) continue

    for (let w = 3; w < words; w++) {
      s = row[w]
      u = s & masks[base + w]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[w] = sum | (s & ~u)
    }
  }

  let count = 0
  for (let w = 0; w < words; w++) count += popcount(~row[w])
  return count
}

/** Four-word held-pattern LCS for `partialRatio` and prepared scorers. */
function lcsFourWordsPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  const stringText = typeof text === 'string'
  let s0 = -1
  let s1 = -1
  let s2 = -1
  let s3 = -1

  for (let i = 0; i < textLength; i++) {
    let base = -1

    if (stringText) {
      const symbol = text.charCodeAt(textStart + i)
      if (symbol < DIRECT_LOOKUP_LIMIT) {
        base = symbol * 4
      } else {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 4
            : (wideOffsets.get(symbol) ?? -1)
      }
    } else {
      const symbol = text[textStart + i]
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        base = symbol * 4
      } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 4
            : (wideOffsets.get(symbol) ?? -1)
      } else if (symbol === symbol) {
        base = wideOffsets.get(symbol) ?? -1
      }
    }
    if (base < 0) continue

    let u = s0 & masks[base]
    let sum = (s0 + u) | 0
    let carry = ((s0 & u) | ((s0 | u) & ~sum)) >>> 31
    s0 = sum | (s0 & ~u)

    u = s1 & masks[base + 1]
    sum = (s1 + u + carry) | 0
    carry = ((s1 & u) | ((s1 | u) & ~sum)) >>> 31
    s1 = sum | (s1 & ~u)

    u = s2 & masks[base + 2]
    sum = (s2 + u + carry) | 0
    carry = ((s2 & u) | ((s2 | u) & ~sum)) >>> 31
    s2 = sum | (s2 & ~u)

    u = s3 & masks[base + 3]
    sum = (s3 + u + carry) | 0
    s3 = sum | (s3 & ~u)
  }

  return popcount(~s0) + popcount(~s1) + popcount(~s2) + popcount(~s3)
}

/**
 * {@link lcsLengthPreparedBounded} over the words the target can reach.
 *
 * The same Ukkonen band as {@link lcsManyWordsBanded}, against masks that are
 * held rather than built. A subsequence of `required` elements can leave at
 * most `patternLength - required` of the pattern and `textLength - required` of
 * the text unmatched, so at text position `i` no pattern position above
 * `i + 1 + left` has been reached yet and none below `i - right` can still be,
 * and the words outside that window need not be touched. Words the window has
 * passed keep what they were left holding; words it has not reached are still
 * all-ones and count for nothing, which is why the total at the end can be
 * taken over the whole row without checking where the window got to.
 *
 * This is the difference the held pattern used to give away. Trimming a common
 * affix is what narrows the *unprepared* kernel, and a held pattern cannot be
 * trimmed — its masks are built once for a length that every candidate would
 * shorten differently. Without a band there was nothing left to narrow it with,
 * so a 1024-element query read all thirty-two of its words for every candidate
 * whatever the cutoff said; at a cutoff of 90 the band is eight.
 *
 * The periodic reachability check the full-width kernel makes is kept, over the
 * words the window has reached rather than the whole row. Upstream's blockwise
 * kernel has no such check — it applies its cutoff once at the end — and
 * dropping it here looked reasonable, since the band already encodes the cutoff
 * and the bound costs something on every candidate that passes. Measured, that
 * is the wrong trade by a wide margin: without it a rejected candidate scans its
 * whole band rather than the first few rows of it. Banding alone was 4.01x on
 * 1024-element candidates that clear the cutoff and 1.83x at 256, but **0.55x**
 * on candidates that do not — and `extract` under a cutoff is mostly candidates
 * that do not.
 *
 * Every sixteenth row rather than the full-width kernel's every eighth. Swept at
 * 8, 16 and 32 over both lists: 16 is the only one that is not worst at
 * something. At 8 the bound runs often enough to cost the passing candidates
 * (1.61x and 3.03x, against 1.72x and 3.35x at 16); at 32 it runs too rarely to
 * catch the failing 256-element ones (1.59x, against 2.01x). The spread between
 * them is inside `--quick`'s band, so 16 is chosen for being nowhere near last
 * rather than for winning by a measured margin.
 */
function lcsPreparedBanded(
  prepared: PatternMask,
  words: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  const left = prepared.length - required
  const right = textLength - required
  let firstWord = 0
  let lastWord = Math.min(words, Math.ceil((left + 1) / WORD_BITS))

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    // An element the pattern does not hold reproduces every word it touches, so
    // the window can be skipped rather than walked — as in `lcsManyWordsStamped`.
    if (base >= 0) {
      let carry = 0

      for (let word = firstWord; word < lastWord; word++) {
        const s = row[word]
        const u = s & masks[base + word]
        const sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[word] = sum | (s & ~u)
      }
    }

    // The same abandonment the full-width kernel makes, over the words the
    // window has reached: everything above `lastWord` is still all-ones and
    // adds nothing, so the whole row need not be read to bound it.
    if ((i & 15) === 15) {
      let possible = textLength - i - 1
      for (let word = 0; word < lastWord; word++) possible += popcount(~row[word])
      if (possible < required) return -1
    }

    if (i > right) firstWord = (i - right) >>> WORD_SHIFT
    // See {@link lcsManyWordsBanded}: the word holding position `i + 1 + left`,
    // not the count of words below it.
    if (i + 1 + left <= prepared.length) {
      lastWord = Math.min(words, ((i + 1 + left) >>> WORD_SHIFT) + 1)
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  // Only a count that reached the target is trustworthy: the band was drawn
  // around alignments that reach it, so anything shorter may have been cut off
  // by the window rather than by the inputs. The caller has already rejected
  // the pair either way.
  return count >= required ? count : -1
}

/**
 * Prepared LCS with a conservative acceptance bound.
 *
 * A negative result only means the requested LCS can no longer be reached;
 * every non-negative result is the exact length.
 */
export function lcsLengthPreparedBounded(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  required: number,
): number {
  const words = prepared.words
  if (words === 0 || textLength === 0) return required > 0 ? -1 : 0

  // A subsequence common to both cannot be longer than either, so a target
  // above one of them is out of reach before a single element is read. This is
  // upstream's `score_cutoff > len1 || score_cutoff > len2`, and the banded
  // branch below relies on it: it is what keeps both band widths non-negative.
  if (required > prepared.length || required > textLength) return -1

  if (words === 1) {
    let s = -1
    const masks = prepared.masks
    const highBase = prepared.highBase
    const highCount = prepared.highCount
    const highStart = prepared.highStart
    const wideOffsets = prepared.wideOffsets

    // As in {@link lcsLengthPrepared}: a string settles every per-element test
    // in advance, and this is the bounded form the prepared `process` paths
    // reach.
    if (typeof text === 'string') {
      for (let i = 0; i < textLength; i++) {
        const symbol = text.charCodeAt(textStart + i)
        let base: number
        if (symbol < DIRECT_LOOKUP_LIMIT) {
          base = symbol
        } else {
          const shifted = symbol - highBase
          base =
            shifted >= 0 && shifted < highCount
              ? highStart + shifted
              : (wideOffsets.get(symbol) ?? -1)
        }
        const matches = base < 0 ? 0 : masks[base]
        const u = s & matches
        s = (s + u) | 0 | (s & ~u)
        if (popcount(~s) + textLength - i - 1 < required) return -1
      }
      return popcount(~s)
    }

    for (let i = 0; i < textLength; i++) {
      const symbol = text[textStart + i]
      // Written out rather than called, and that is load bearing — see the note
      // on `patternBase`, whose body this is. A single shared copy sees numbers
      // from string inputs and strings and objects from array inputs, goes
      // megamorphic, and measured 2.43x slower once other kernels had used it.
      let base = -1
      if (
        typeof symbol === 'number' &&
        symbol >= 0 &&
        symbol < DIRECT_LOOKUP_LIMIT &&
        (symbol | 0) === symbol
      ) {
        base = symbol * 1
      } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
        const shifted = symbol - highBase
        base =
          shifted >= 0 && shifted < highCount
            ? highStart + shifted * 1
            : (wideOffsets.get(symbol) ?? -1)
      } else if (symbol === symbol) {
        base = wideOffsets.get(symbol) ?? -1
      }
      const matches = base < 0 ? 0 : masks[base]
      const u = s & matches
      s = (s + u) | 0 | (s & ~u)
      if (popcount(~s) + textLength - i - 1 < required) return -1
    }
    return popcount(~s)
  }

  // How far the alignment can drift from the diagonal, in words. The `+ 2`
  // covers the two partial words the band's ends fall inside; below `words` it
  // is the whole reason to prefer the banded kernel, and at `words` there is
  // nothing to skip, so the full-width loop below is the cheaper way to say so.
  // A target of zero rejects nothing and draws no band worth having.
  const fullBand = prepared.length + textLength - 2 * required + 1
  const bandWords = Math.min(words, Math.floor(fullBand / WORD_BITS) + 2)
  if (required > 0 && bandWords < words) {
    return lcsPreparedBanded(prepared, words, text, textStart, textLength, required)
  }

  const row = rowVector(words)
  clearRange(row, -1, 0, words)
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets
  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
    // As in `lcsLengthPrepared`: where the masks start is settled once per
    // element, and an element the pattern does not hold leaves the row alone.
    // The bound below still has to run — it reads the row rather than
    // advancing it.
    // Written out rather than called, and that is load bearing — see the note
    // on `patternBase`, whose body this is. A single shared copy sees numbers
    // from string inputs and strings and objects from array inputs, goes
    // megamorphic, and measured 2.43x slower once other kernels had used it.
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < DIRECT_LOOKUP_LIMIT &&
      (symbol | 0) === symbol
    ) {
      base = symbol * words
    } else if (typeof symbol === 'number' && (symbol | 0) === symbol) {
      const shifted = symbol - highBase
      base =
        shifted >= 0 && shifted < highCount
          ? highStart + shifted * words
          : (wideOffsets.get(symbol) ?? -1)
    } else if (symbol === symbol) {
      base = wideOffsets.get(symbol) ?? -1
    }

    if (base >= 0) {
      // Written out to three words for the reason given in `lcsLengthPrepared`.
      let s = row[0]
      let u = s & masks[base]
      let sum = (s + u) | 0
      let carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[0] = sum | (s & ~u)

      s = row[1]
      u = s & masks[base + 1]
      sum = (s + u + carry) | 0
      carry = ((s & u) | ((s | u) & ~sum)) >>> 31
      row[1] = sum | (s & ~u)

      if (words > 2) {
        s = row[2]
        u = s & masks[base + 2]
        sum = (s + u + carry) | 0
        carry = ((s & u) | ((s | u) & ~sum)) >>> 31
        row[2] = sum | (s & ~u)

        for (let word = 3; word < words; word++) {
          s = row[word]
          u = s & masks[base + word]
          sum = (s + u + carry) | 0
          carry = ((s & u) | ((s | u) & ~sum)) >>> 31
          row[word] = sum | (s & ~u)
        }
      }
    }

    if ((i & 7) === 7 || i + 1 === textLength) {
      let possible = textLength - i - 1
      for (let word = 0; word < words; word++) possible += popcount(~row[word])
      if (possible < required) return -1
    }
  }

  let count = 0
  for (let word = 0; word < words; word++) count += popcount(~row[word])
  return count
}

/** Exact length of the longest common subsequence. */
export function lcsLength(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return lcsLengthRange(s1, 0, s1.length, s2, 0, s2.length, UNBOUNDED_MISSES)
}
