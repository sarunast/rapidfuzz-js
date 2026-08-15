import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { checkedStartGeneration } from '../shared/bitmask/blockMasks.js'
import {
  alignRepresentation,
  canonicalRawCutoff,
  canonicalSimilarityCutoff,
  convPair,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type ScorerOptions,
  type Sequence,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MetricImplementation,
} from '../shared/scorerSupport.js'

let rowA: Int32Array | null = null
let rowB: Int32Array | null = null
let transposeRow: Int32Array | null = null
let shortRowA: Int16Array | null = null
let shortRowB: Int16Array | null = null
let shortTransposeRow: Int16Array | null = null
let bmpLastRow: Int32Array | null = null
let bmpLastStamp: Int32Array | null = null
let bmpGeneration = 0

function bmpLastRows(): Int32Array {
  return (bmpLastRow ??= new Int32Array(0x10000))
}

function bmpLastStamps(): Int32Array {
  return (bmpLastStamp ??= new Int32Array(0x10000))
}

export function resetDamerauScratch(startGeneration = 0): void {
  rowA = null
  rowB = null
  transposeRow = null
  shortRowA = null
  shortRowB = null
  shortTransposeRow = null
  bmpLastRow = null
  bmpLastStamp = null
  bmpGeneration = checkedStartGeneration(startGeneration)
}

function grown(buffer: Int32Array | null, needed: number): Int32Array {
  if (buffer !== null && buffer.length >= needed) return buffer
  let size = buffer === null ? 64 : buffer.length
  while (size < needed) size *= 2
  return new Int32Array(size)
}

function grownShort(buffer: Int16Array | null, needed: number): Int16Array {
  if (buffer !== null && buffer.length >= needed) return buffer
  let size = buffer === null ? 64 : buffer.length
  while (size < needed) size *= 2
  return new Int16Array(size)
}

function isBmpCode(value: unknown): value is number {
  return (
    typeof value === 'number' && value >= 0 && value <= 0xffff && (value | 0) === value
  )
}

const BMP_GENERATION_LIMIT = 0x7fff_ffff

function nextBmpGeneration(): number {
  bmpGeneration++
  if (bmpGeneration >= BMP_GENERATION_LIMIT) {
    bmpLastStamps().fill(0)
    bmpGeneration = 1
  }
  return bmpGeneration
}

function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff = Number.MAX_SAFE_INTEGER,
): number {
  const fullLen1 = s1.length
  const fullLen2 = s2.length

  if (fullLen1 === 0) return fullLen2
  if (fullLen2 === 0) return fullLen1
  if (Math.abs(fullLen1 - fullLen2) > scoreCutoff) return scoreCutoff + 1

  let prefix = 0
  const shorter = Math.min(fullLen1, fullLen2)
  while (prefix < shorter && s1[prefix] === s2[prefix]) prefix++
  let suffix = 0
  while (
    suffix < shorter - prefix &&
    s1[fullLen1 - suffix - 1] === s2[fullLen2 - suffix - 1]
  ) {
    suffix++
  }

  const len1 = fullLen1 - prefix - suffix
  const len2 = fullLen2 - prefix - suffix
  if (len1 === 0) return len2
  if (len2 === 0) return len1

  const maxValue = Math.max(len1, len2) + 1
  let previous: Int16Array | Int32Array
  let current: Int16Array | Int32Array
  let transpositionRows: Int16Array | Int32Array
  if (maxValue <= 0x7fff) {
    shortRowA = grownShort(shortRowA, len2 + 2)
    shortRowB = grownShort(shortRowB, len2 + 2)
    shortTransposeRow = grownShort(shortTransposeRow, len2 + 2)
    previous = shortRowA
    current = shortRowB
    transpositionRows = shortTransposeRow
  } else {
    rowA = grown(rowA, len2 + 2)
    rowB = grown(rowB, len2 + 2)
    transposeRow = grown(transposeRow, len2 + 2)
    previous = rowA
    current = rowB
    transpositionRows = transposeRow
  }
  previous.fill(maxValue, 0, len2 + 2)
  transpositionRows.fill(maxValue, 0, len2 + 2)
  current[0] = maxValue
  for (let j = 0; j <= len2; j++) current[j + 1] = j

  const stampedRow = bmpLastRows()
  const stamps = bmpLastStamps()
  const directGeneration = nextBmpGeneration()

  if (typeof s1 === 'string' && typeof s2 === 'string') {
    for (let i = 1; i <= len1; i++) {
      const spare = previous
      previous = current
      current = spare
      let lastColumn = -1
      let lastI2L1 = current[1]
      current[1] = i
      let diagonalTranspose = maxValue
      const a = s1.charCodeAt(prefix + i - 1)

      for (let j = 1; j <= len2; j++) {
        const b = s2.charCodeAt(prefix + j - 1)
        let value = Math.min(
          previous[j] + (a === b ? 0 : 1),
          current[j] + 1,
          previous[j + 1] + 1,
        )

        if (a === b) {
          lastColumn = j
          transpositionRows[j + 1] = previous[j - 1]
          diagonalTranspose = lastI2L1
        } else {
          const last = stamps[b] === directGeneration ? stampedRow[b] : -1
          if (j - lastColumn === 1) {
            value = Math.min(value, transpositionRows[j + 1] + (i - last))
          } else if (i - last === 1) {
            value = Math.min(value, diagonalTranspose + (j - lastColumn))
          }
        }

        lastI2L1 = current[j + 1]
        current[j + 1] = value
      }

      stampedRow[a] = i
      stamps[a] = directGeneration
    }

    const resultString = current[len2 + 1]
    return resultString <= scoreCutoff ? resultString : scoreCutoff + 1
  }

  let lastRow: Map<unknown, number> | null = null

  for (let i = 1; i <= len1; i++) {
    const spare = previous
    previous = current
    current = spare
    let lastColumn = -1
    let lastI2L1 = current[1]
    current[1] = i
    let diagonalTranspose = maxValue
    const a = s1[prefix + i - 1]

    for (let j = 1; j <= len2; j++) {
      const b = s2[prefix + j - 1]
      let value = Math.min(
        previous[j] + (a === b ? 0 : 1),
        current[j] + 1,
        previous[j + 1] + 1,
      )

      if (a === b) {
        lastColumn = j
        transpositionRows[j + 1] = previous[j - 1]
        diagonalTranspose = lastI2L1
      } else {
        let last = -1
        if (typeof b === 'number' && b >= 0 && b <= 0xffff && (b | 0) === b) {
          if (stamps[b] === directGeneration) last = stampedRow[b]
        } else if (lastRow !== null && b === b) {
          last = lastRow.get(b) ?? -1
        }
        if (j - lastColumn === 1) {
          value = Math.min(value, transpositionRows[j + 1] + (i - last))
        } else if (i - last === 1) {
          value = Math.min(value, diagonalTranspose + (j - lastColumn))
        }
      }

      lastI2L1 = current[j + 1]
      current[j + 1] = value
    }

    if (isBmpCode(a)) {
      stampedRow[a] = i
      stamps[a] = directGeneration
    } else if (a === a) {
      if (lastRow === null) lastRow = new Map<unknown, number>()
      lastRow.set(a, i)
    }
  }

  const result = current[len2 + 1]
  return result <= scoreCutoff ? result : scoreCutoff + 1
}

type PreparedDamerauKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareDamerau(kind: PreparedDamerauKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      let b = preparedChoiceSequence(rawChoice)
      const s1 = alignRepresentation(a, b)
      b = alignRepresentation(b, a)
      const max = maximum(s1, b)
      switch (kind) {
        case 'distance': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          return distCutoff(distance_(s1, b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
        }
        case 'similarity': {
          const cutoff = canonicalSimilarityCutoff(rawCutoff)
          const bound = cutoff === null ? Number.MAX_SAFE_INTEGER : max - cutoff
          return simCutoff(max - distance_(s1, b, bound), cutoff)
        }
        case 'normalizedDistance': {
          const cutoff = rawCutoff === null ? Number.MAX_SAFE_INTEGER : rawCutoff * max
          return normDistCutoff(
            normalizeDistance(distance_(s1, b, cutoff), max),
            rawCutoff,
          )
        }
        case 'normalizedSimilarity': {
          const cutoff =
            rawCutoff === null ? Number.MAX_SAFE_INTEGER : (1 - rawCutoff) * max
          return normSimCutoff(
            1 - normalizeDistance(distance_(s1, b, cutoff), max),
            rawCutoff,
          )
        }
      }
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function damerauLevenshteinDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  return distCutoff(distance_(a, b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
}

function damerauLevenshteinSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = canonicalSimilarityCutoff(options.scoreCutoff)
  const bound = cutoff == null ? Number.MAX_SAFE_INTEGER : max - cutoff
  return simCutoff(max - distance_(a, b, bound), cutoff)
}

function damerauLevenshteinNormalizedDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? Number.MAX_SAFE_INTEGER : options.scoreCutoff * max
  return normDistCutoff(
    normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

function damerauLevenshteinNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : (1 - options.scoreCutoff) * max
  return normSimCutoff(
    1 - normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

export const damerauLevenshteinDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauLevenshteinDistance_impl,
    DISTANCE_FLAGS,
    prepareDamerau('distance'),
  )
export const damerauLevenshteinSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauLevenshteinSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareDamerau('similarity'),
  )
export const damerauLevenshteinNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauLevenshteinNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareDamerau('normalizedDistance'),
  )
export const damerauLevenshteinNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauLevenshteinNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareDamerau('normalizedSimilarity'),
  )
