/**
 * Shared Optimal String Alignment: Myers' recurrence plus the `TR` term that admits a
 * transposition of two adjacent elements.
 *
 * Separate from `levenshtein.ts` because `TR` reaches back into the previous
 * column, which needs a second carry chain the LCS and Levenshtein kernels do
 * not have. That extra chain is also why this module owns {@link osaScratch}
 * outright rather than sharing the row vectors in `shared.ts`.
 */

import {
  directSlots,
  directStamps,
  buildWordMasks,
  directLimit,
  wideSlots,
} from '../../shared/bitmask/blockMasks.js'
import type { PatternMask } from '../../shared/bitmask/pattern.js'

// Declared here rather than imported — see the note in `shared.ts`, which holds
// the canonical definitions. Read once per element, where a cross-module
// binding does not fold the way a module-local `const` does.
const WORD_BITS = 32
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

/**
 * Optimal String Alignment over one word — port of `_osa_distance_hyrroe2003`.
 *
 * Myers' recurrence plus the `TR` term, which is what admits a transposition of
 * two adjacent elements. `TR` reaches back into the previous column, so
 * carrying it across words needs a second chain the LCS and Levenshtein kernels
 * do not have — {@link osaPrepared} keeps it, for patterns past one word.
 */
export function osaOneWord(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
): number {
  return osaOneWordRange(pattern, 0, pattern.length, text, 0, text.length)
}

/** Single-word OSA over caller-validated ranges. */
export function osaOneWordRange(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  if (patternLength === 0) return textLength
  // Every shift below is taken modulo 32, so a longer pattern would not fail —
  // it would quietly wrap and return a number that means nothing.
  if (patternLength > WORD_BITS) {
    throw new RangeError(`osaOneWord supports at most ${WORD_BITS} elements`)
  }
  if (textLength === 0) return patternLength

  const stamp = buildWordMasks(pattern, patternStart, patternLength)
  const slots = directSlots()
  const stamps = directStamps()
  const wide = wideSlots()

  const top = 1 << (patternLength - 1)

  let vp = -1
  let vn = 0
  let d0 = 0
  let previousMatches = 0
  let distance = patternLength
  const limit = directLimit
  const stringText = typeof text === 'string'

  for (let i = 0; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(textStart + i) : text[textStart + i]
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

    // A transposition is available exactly where this column matches and the
    // previous one matched the other way round.
    const transposed = ((~d0 & matches) << 1) & previousMatches
    d0 = (((matches & vp) + vp) ^ vp) | matches | vn | transposed | 0

    let hp = vn | ~(d0 | vp)
    let hn = d0 & vp

    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--

    hp = (hp << 1) | 1
    hn = hn << 1
    vp = hn | ~(d0 | hp)
    vn = hp & d0
    previousMatches = matches
  }

  return distance
}

/** Single-word OSA against immutable query masks. */
export function osaOneWordPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number = 0,
  textLength: number = text.length - textStart,
): number {
  const patternLength = prepared.length
  if (patternLength === 0) return textLength
  // As in `osaOneWordRange`: past one word the shifts wrap instead of failing.
  if (patternLength > WORD_BITS) {
    throw new RangeError(`osaOneWordPrepared supports at most ${WORD_BITS} elements`)
  }
  if (textLength === 0) return patternLength
  const top = 1 << (patternLength - 1)
  let vp = -1
  let vn = 0
  let d0 = 0
  let previousMatches = 0
  let distance = patternLength
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

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

    const transposed = ((~d0 & matches) << 1) & previousMatches
    d0 = (((matches & vp) + vp) ^ vp) | matches | vn | transposed | 0
    let hp = vn | ~(d0 | vp)
    let hn = d0 & vp
    if ((hp & top) !== 0) distance++
    if ((hn & top) !== 0) distance--
    hp = (hp << 1) | 1
    hn <<= 1
    vp = hn | ~(d0 | hp)
    vn = hp & d0
    previousMatches = matches
  }

  return distance
}

/** OSA against immutable query masks for repeated process scoring. */
let osaScratch = new Int32Array(0)

/**
 * The eight rolling vectors, as views onto {@link osaScratch}.
 *
 * Eight `subarray` calls is eight object allocations per scored pair, and a
 * `process` row scores one pair per choice — so they are cached across calls.
 *
 * The cache key is `stride`, not whether the backing array had to grow. Those
 * are different questions: the scratch is grown to a power of two and never
 * shrinks, so a later query with a *shorter* pattern leaves it untouched while
 * moving all eight boundaries. Keying on the allocation would hand that query
 * the previous query's windows, which overlap each other and silently corrupt
 * the recurrence. `osaViewStride` starts at a width no `stride` can take —
 * `stride` is `words + 1`, and `words === 0` returns before this point — so the
 * first call builds them.
 */
let osaViewStride = 0
let osaViewVp = new Int32Array(0)
let osaViewVn = new Int32Array(0)
let osaViewD0 = new Int32Array(0)
let osaViewPm = new Int32Array(0)
let osaViewVp2 = new Int32Array(0)
let osaViewVn2 = new Int32Array(0)
let osaViewD02 = new Int32Array(0)
let osaViewPm2 = new Int32Array(0)

export function osaPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number = 0,
  textLength: number = text.length - textStart,
): number {
  const words = prepared.words
  if (words === 0) return textLength
  if (textLength === 0) return prepared.length
  const stride = words + 1
  const needed = stride * 8
  if (osaScratch.length < needed) {
    let size = Math.max(64, osaScratch.length)
    while (size < needed) size *= 2
    osaScratch = new Int32Array(size)
    // The views point into the array that was just replaced.
    osaViewStride = 0
  }
  if (osaViewStride !== stride) {
    osaViewStride = stride
    osaViewVp = osaScratch.subarray(0, stride)
    osaViewVn = osaScratch.subarray(stride, stride * 2)
    osaViewD0 = osaScratch.subarray(stride * 2, stride * 3)
    osaViewPm = osaScratch.subarray(stride * 3, stride * 4)
    osaViewVp2 = osaScratch.subarray(stride * 4, stride * 5)
    osaViewVn2 = osaScratch.subarray(stride * 5, stride * 6)
    osaViewD02 = osaScratch.subarray(stride * 6, stride * 7)
    osaViewPm2 = osaScratch.subarray(stride * 7, stride * 8)
  }
  const oldVp = osaViewVp
  const oldVn = osaViewVn
  const oldD0 = osaViewD0
  const oldPm = osaViewPm
  const newVp = osaViewVp2
  const newVn = osaViewVn2
  const newD0 = osaViewD02
  const newPm = osaViewPm2
  osaScratch.fill(0, 0, needed)
  oldVp.fill(-1)
  newVp.fill(-1)
  const last = 1 << ((prepared.length - 1) & WORD_MASK)
  let distance = prepared.length

  let sourceVp = oldVp
  let sourceVn = oldVn
  let sourceD0 = oldD0
  let sourcePm = oldPm
  let targetVp = newVp
  let targetVn = newVn
  let targetD0 = newD0
  let targetPm = newPm
  const stringText = typeof text === 'string'
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let row = 0; row < textLength; row++) {
    let hpCarry = 1
    let hnCarry = 0
    const symbol = stringText ? text.charCodeAt(textStart + row) : text[textStart + row]
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

    for (let word = 0; word < words; word++) {
      const vn = sourceVn[word + 1]
      const vp = sourceVp[word + 1]
      let d0 = sourceD0[word + 1]
      const d0Last = sourceD0[word]
      const previousMatches = sourcePm[word + 1]
      const matchesLastWord = targetPm[word]
      const matches = base < 0 ? 0 : masks[base + word]

      const transposed =
        (((~d0 & matches) << 1) | ((~d0Last & matchesLastWord) >>> 31)) & previousMatches
      const x = matches | hnCarry
      d0 = (((x & vp) + vp) ^ vp) | x | vn | transposed | 0
      let hp = vn | ~(d0 | vp)
      let hn = d0 & vp

      if (word === words - 1) {
        if ((hp & last) !== 0) distance++
        if ((hn & last) !== 0) distance--
      }

      const nextHpCarry = hp >>> 31
      const nextHnCarry = hn >>> 31
      hp = (hp << 1) | hpCarry
      hn = (hn << 1) | hnCarry
      hpCarry = nextHpCarry
      hnCarry = nextHnCarry

      targetVp[word + 1] = hn | ~(d0 | hp)
      targetVn[word + 1] = hp & d0
      targetD0[word + 1] = d0
      targetPm[word + 1] = matches
    }

    let swap = sourceVp
    sourceVp = targetVp
    targetVp = swap
    swap = sourceVn
    sourceVn = targetVn
    targetVn = swap
    swap = sourceD0
    sourceD0 = targetD0
    targetD0 = swap
    swap = sourcePm
    sourcePm = targetPm
    targetPm = swap
  }

  return distance
}

/** Benchmark/test seam that returns retained scratch to a deterministic baseline. */
export function resetOsaScratch(): void {
  // The eight views have to be dropped with it: a `subarray` keeps the whole
  // backing buffer alive, so replacing `osaScratch` alone frees nothing until
  // the next call rebuilds them.
  const empty = new Int32Array(0)
  osaScratch = empty
  osaViewStride = 0
  osaViewVp = empty
  osaViewVn = empty
  osaViewD0 = empty
  osaViewPm = empty
  osaViewVp2 = empty
  osaViewVn2 = empty
  osaViewD02 = empty
  osaViewPm2 = empty
}

/** Test seam: the largest buffer any binding in this module still holds. */
export function osaRetainedBytes(): number {
  return Math.max(
    osaScratch.buffer.byteLength,
    osaViewVp.buffer.byteLength,
    osaViewVn.buffer.byteLength,
    osaViewD0.buffer.byteLength,
    osaViewPm.buffer.byteLength,
    osaViewVp2.buffer.byteLength,
    osaViewVn2.buffer.byteLength,
    osaViewD02.buffer.byteLength,
    osaViewPm2.buffer.byteLength,
  )
}
