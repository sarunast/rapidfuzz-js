import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { wordCount } from '../shared/bitmask/blockMasks.js'
import { preparePattern, type PatternMask } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  validateSequence,
  convPair,
  normDistCutoff,
  normSimCutoff,
  type MaybeSequence,
  type Sequence,
  type MaybeSequenceMetricImplementation,
  type ScorerOptions,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  withPreparedFlags,
} from '../shared/scorerSupport.js'

let patternFlags: Uint32Array | null = null
let textFlags: Uint32Array | null = null

export function resetJaroScratch(): void {
  patternFlags = null
  textFlags = null
}

function grown(buffer: Uint32Array | null, needed: number): Uint32Array {
  if (buffer !== null && buffer.length >= needed) return buffer
  let size = buffer === null ? 32 : buffer.length
  while (size < needed) size *= 2
  return new Uint32Array(size)
}

export function jaroSimilarity_(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  scoreCutoff = 0,
): number {
  return jaroSimilarityCore(pattern, text, scoreCutoff)
}

export function jaroSimilarityPrepared_(
  pattern: ArrayLike<unknown>,
  preparedPattern: PatternMask,
  text: ArrayLike<unknown>,
  scoreCutoff = 0,
): number {
  return jaroSimilarityCore(pattern, text, scoreCutoff, preparedPattern)
}

function jaroSimilarityCore(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  scoreCutoff: number,
  preparedPattern?: PatternMask,
): number {
  const lenP = pattern.length
  const lenT = text.length

  if (lenP === 0 && lenT === 0) return 1
  if (lenP === 0 || lenT === 0) return 0

  const minLength = Math.min(lenP, lenT)
  const ceiling = (minLength / lenP + minLength / lenT + 1) / 3
  if (ceiling < scoreCutoff) return 0

  const bound = Math.max(Math.floor(Math.max(lenP, lenT) / 2) - 1, 0)
  let patternEnd = lenP
  let textEnd = lenT
  if (lenT > lenP && lenT > lenP + bound) textEnd = lenP + bound
  else if (lenP > lenT && lenP > lenT + bound) patternEnd = lenT + bound
  let prefix = 0
  const activeShorter = Math.min(patternEnd, textEnd)
  if (typeof pattern === 'string' && typeof text === 'string') {
    while (
      prefix < activeShorter &&
      pattern.charCodeAt(prefix) === text.charCodeAt(prefix)
    ) {
      prefix++
    }
  } else {
    while (prefix < activeShorter && pattern[prefix] === text[prefix]) prefix++
  }

  const origin = preparedPattern === undefined ? prefix : 0
  const skip = prefix - origin
  const patternLength = patternEnd - origin
  const textLength = textEnd - origin
  const prepared = preparedPattern ?? preparePattern(pattern, prefix, patternLength)
  const patternWords = Math.max(wordCount(patternLength), 1)
  const textWords = Math.max(wordCount(textLength), 1)
  if (patternWords === 1 && textWords === 1) {
    return jaroOneWord(
      pattern,
      text,
      origin,
      prefix,
      patternLength,
      textLength,
      bound,
      prepared,
      lenP,
      lenT,
      scoreCutoff,
    )
  }
  patternFlags = grown(patternFlags, patternWords)
  textFlags = grown(textFlags, textWords)
  const pFlag = patternFlags
  const tFlag = textFlags
  pFlag.fill(0, 0, patternWords)
  tFlag.fill(0, 0, textWords)
  if (skip > 0) {
    const wholeWords = skip >>> 5
    pFlag.fill(0xffff_ffff, 0, wholeWords)
    const remainder = skip & 31
    if (remainder !== 0) pFlag[wholeWords] = (1 << remainder) - 1
  }
  let common = prefix
  const stringText = typeof text === 'string'
  const words = prepared.words
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let i = skip; i < textLength; i++) {
    const low = Math.max(0, i - bound)
    const high = Math.min(patternLength - 1, i + bound)
    const firstWord = low >>> 5
    const lastWord = high >>> 5
    const symbol = stringText ? text.charCodeAt(origin + i) : text[origin + i]
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < 256 &&
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

    for (let word = firstWord; word <= lastWord; word++) {
      const matches = base < 0 ? 0 : masks[base + word]
      let available = matches & ~pFlag[word]
      if (word === firstWord) available &= -1 << (low & 31)
      if (word === lastWord && (high & 31) !== 31) {
        available &= (1 << ((high & 31) + 1)) - 1
      }
      if (available === 0) continue

      pFlag[word] |= available & -available
      tFlag[i >>> 5] |= 1 << (i & 31)
      common++
      break
    }
  }

  if (common === 0) return 0

  const commonCeiling = (common / lenP + common / lenT + 1) / 3
  if (commonCeiling < scoreCutoff) return 0

  const transpositions = countTranspositionsWords(
    pattern,
    text,
    origin,
    skip,
    textLength,
    pFlag,
    tFlag,
  )

  return finishJaro(lenP, lenT, common, transpositions, scoreCutoff)
}

function countTranspositionsWords(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  skip: number,
  textLength: number,
  pFlag: Uint32Array,
  tFlag: Uint32Array,
): number {
  const textWords = wordCount(textLength)
  const firstWord = skip >>> 5
  const leading = -1 << (skip & 31)

  let transpositions = 0
  let patternWord = firstWord
  let patternBits = pFlag[firstWord] & leading

  if (typeof pattern === 'string' && typeof text === 'string') {
    for (let word = firstWord; word < textWords; word++) {
      let bits = word === firstWord ? tFlag[word] & leading : tFlag[word]

      while (bits !== 0) {
        const lowest = bits & -bits
        bits ^= lowest
        const i = (word << 5) + 31 - Math.clz32(lowest)

        while (patternBits === 0) {
          patternWord++
          patternBits = pFlag[patternWord]
        }
        const patternLowest = patternBits & -patternBits
        patternBits ^= patternLowest
        const patternIndex = (patternWord << 5) + 31 - Math.clz32(patternLowest)

        if (text.charCodeAt(origin + i) !== pattern.charCodeAt(origin + patternIndex)) {
          transpositions++
        }
      }
    }

    return transpositions
  }

  for (let word = firstWord; word < textWords; word++) {
    let bits = word === firstWord ? tFlag[word] & leading : tFlag[word]

    while (bits !== 0) {
      const lowest = bits & -bits
      bits ^= lowest
      const i = (word << 5) + 31 - Math.clz32(lowest)

      while (patternBits === 0) {
        patternWord++
        patternBits = pFlag[patternWord]
      }
      const patternLowest = patternBits & -patternBits
      patternBits ^= patternLowest
      const patternIndex = (patternWord << 5) + 31 - Math.clz32(patternLowest)

      if (text[origin + i] !== pattern[origin + patternIndex]) transpositions++
    }
  }

  return transpositions
}

function countTranspositionsOneWord(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  skip: number,
  patternFlag: number,
  textFlag: number,
): number {
  const leading = -1 << (skip & 31)
  let transpositions = 0
  let bits = textFlag & leading
  let patternBits = patternFlag & leading

  if (typeof pattern === 'string' && typeof text === 'string') {
    while (bits !== 0 && patternBits !== 0) {
      const lowest = bits & -bits
      const patternLowest = patternBits & -patternBits
      bits ^= lowest
      patternBits ^= patternLowest

      if (
        text.charCodeAt(origin + 31 - Math.clz32(lowest)) !==
        pattern.charCodeAt(origin + 31 - Math.clz32(patternLowest))
      ) {
        transpositions++
      }
    }

    return transpositions
  }

  while (bits !== 0 && patternBits !== 0) {
    const lowest = bits & -bits
    const patternLowest = patternBits & -patternBits
    bits ^= lowest
    patternBits ^= patternLowest

    if (
      text[origin + 31 - Math.clz32(lowest)] !==
      pattern[origin + 31 - Math.clz32(patternLowest)]
    ) {
      transpositions++
    }
  }

  return transpositions
}

function finishJaro(
  patternLength: number,
  textLength: number,
  common: number,
  transpositions: number,
  scoreCutoff: number,
): number {
  const halfTranspositions = Math.floor(transpositions / 2)
  const result =
    (common / patternLength +
      common / textLength +
      (common - halfTranspositions) / common) /
    3
  return result >= scoreCutoff ? result : 0
}

function jaroOneWord(
  pattern: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  origin: number,
  prefix: number,
  patternLength: number,
  textLength: number,
  bound: number,
  prepared: PatternMask,
  fullPatternLength: number,
  fullTextLength: number,
  scoreCutoff: number,
): number {
  const skip = prefix - origin
  let patternFlag = skip >= 32 ? -1 : (1 << skip) - 1
  let textFlag = 0
  let common = prefix
  let low = Math.max(0, skip - bound)
  let high = Math.min(patternLength - 1, skip + bound)
  let firstMask = -1 << (low & 31)
  let lastMask = (high & 31) === 31 ? -1 : (1 << ((high & 31) + 1)) - 1
  const stringText = typeof text === 'string'
  const words = prepared.words
  const masks = prepared.masks
  const highBase = prepared.highBase
  const highCount = prepared.highCount
  const highStart = prepared.highStart
  const wideOffsets = prepared.wideOffsets

  for (let i = skip; i < textLength; i++) {
    const symbol = stringText ? text.charCodeAt(origin + i) : text[origin + i]
    let base = -1
    if (
      typeof symbol === 'number' &&
      symbol >= 0 &&
      symbol < 256 &&
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
    const matches = base < 0 ? 0 : masks[base]
    const available = matches & ~patternFlag & firstMask & lastMask
    if (available !== 0) {
      patternFlag |= available & -available
      textFlag |= 1 << i
      common++
    }

    if (i >= bound) {
      low++
      firstMask <<= 1
    }
    if (high + 1 < patternLength) {
      high++
      lastMask = (lastMask << 1) | 1
    }
  }

  if (common === 0) return 0
  const commonCeiling = (common / fullPatternLength + common / fullTextLength + 1) / 3
  if (commonCeiling < scoreCutoff) return 0

  const transpositions = countTranspositionsOneWord(
    pattern,
    text,
    origin,
    skip,
    patternFlag,
    textFlag,
  )

  return finishJaro(
    fullPatternLength,
    fullTextLength,
    common,
    transpositions,
    scoreCutoff,
  )
}

type PreparedJaroKind = 'distance' | 'similarity'

function prepareJaro(kind: PreparedJaroKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)

      const similarityCutoff =
        kind === 'distance' ? (rawCutoff === null ? 0 : 1 - rawCutoff) : (rawCutoff ?? 0)
      const similarity = jaroSimilarityPrepared_(
        alignRepresentation(a, b),
        pattern,
        alignRepresentation(b, a),
        similarityCutoff,
      )
      return kind === 'distance'
        ? normDistCutoff(1 - similarity, rawCutoff)
        : normSimCutoff(similarity, rawCutoff)
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

function jaroSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normSimCutoff(
    jaroSimilarity_(a, b, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

function jaroDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const cutoff = options.scoreCutoff
  return normDistCutoff(
    1 - jaroSimilarity_(a, b, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

export const jaroSimilarity: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    jaroSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaro('similarity'),
  )
export const jaroDistance: MaybeSequenceMetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    jaroDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareJaro('distance'),
  )
