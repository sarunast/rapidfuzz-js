import {
  directSlots,
  directStamps,
  buildWordMasks,
  directLimit,
  wideSlots,
} from '../../shared/bitmask/blockMasks.js'
import type { PatternMask } from '../../shared/bitmask/pattern.js'

const WORD_BITS = 32
const WORD_MASK = 31
const DIRECT_LOOKUP_LIMIT = 256

export function osaOneWordRange(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  if (patternLength === 0) return textLength
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

export function osaOneWordPrepared(
  prepared: PatternMask,
  text: ArrayLike<unknown>,
  textStart: number = 0,
  textLength: number = text.length - textStart,
): number {
  const patternLength = prepared.length
  if (patternLength === 0) return textLength
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

let osaScratch = new Int32Array(0)

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

export function resetOsaScratch(): void {
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
