import {
  distanceCutoffFor,
  scoreFromDistance,
  type MetricScoreKind,
} from '#core/scoring/builtIn/cutoff.js'
import { directMetric } from '#core/scoring/builtIn/directMetric.js'
import {
  withPreparedFlags,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MetricImplementation,
} from '#core/scoring/builtIn/implementation.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  type PreparationFactory,
} from '#core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import {
  alignRepresentation,
  queryAligner,
  scorerSequence,
  maxSequenceLength,
} from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { checkedStartGeneration } from '../bitmask/blockMasks.js'

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

function prepareDamerau(kind: MetricScoreKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    const alignedQueryFor = queryAligner(a)
    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      let b = preparedChoiceSequence(rawChoice)
      const s1 = alignedQueryFor(b)
      b = alignRepresentation(b, a)
      const max = maxSequenceLength(s1, b)
      const budget = distanceCutoffFor(kind, rawCutoff, max, Number.MAX_SAFE_INTEGER)
      return scoreFromDistance(kind, distance_(s1, b, budget), max, rawCutoff)
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

function damerauMetric(kind: MetricScoreKind) {
  return directMetric(kind, distance_, maxSequenceLength, Number.MAX_SAFE_INTEGER)
}

export const damerauLevenshteinDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauMetric('distance'),
    DISTANCE_FLAGS,
    prepareDamerau('distance'),
  )
export const damerauLevenshteinSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauMetric('similarity'),
    SIMILARITY_FLAGS,
    prepareDamerau('similarity'),
  )
export const damerauLevenshteinNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauMetric('normalizedDistance'),
    NORMALIZED_DISTANCE_FLAGS,
    prepareDamerau('normalizedDistance'),
  )
export const damerauLevenshteinNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    damerauMetric('normalizedSimilarity'),
    NORMALIZED_SIMILARITY_FLAGS,
    prepareDamerau('normalizedSimilarity'),
  )
