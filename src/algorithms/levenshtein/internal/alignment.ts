import { preparePattern } from '../../shared/bitmask/pattern.js'
import {
  levenshteinMatrix,
  levenshteinMatrixBytes,
  rowBitSet,
  shiftedRowBitSet,
} from '../../shared/bitParallel.js'
import type { Editop } from '../../shared/editops/index.js'
import {
  ALIGNMENT_MATRIX_LIMIT,
  growHirschbergRows,
  hirschbergLeft,
  hirschbergRight,
} from './scratch.js'
import { levenshteinPreparedRow } from './uniform.js'

function recoveryBitSet(
  rows: Int32Array,
  stride: number,
  offsets: Int32Array | null,
  row: number,
  column: number,
): boolean {
  return offsets === null
    ? rowBitSet(rows, stride, row, column)
    : shiftedRowBitSet(rows, stride, row, offsets[row], column)
}

function recoverMatrixRange(
  out: Editop[],
  source: ArrayLike<unknown>,
  sourceStart: number,
  sourceLength: number,
  destination: ArrayLike<unknown>,
  destinationStart: number,
  destinationLength: number,
  maximumDistance: number,
): void {
  const {
    dist: total,
    vp,
    vn,
    stride,
    offsets,
  } = levenshteinMatrix(
    source,
    sourceStart,
    sourceLength,
    destination,
    destinationStart,
    destinationLength,
    maximumDistance,
  )
  if (total === 0) return
  const rangeOps = new Array<Editop>(total)
  let dist = total
  let col = sourceLength
  let row = destinationLength
  while (row !== 0 && col !== 0) {
    if (recoveryBitSet(vp, stride, offsets, row - 1, col - 1)) {
      dist--
      col--
      rangeOps[dist] = {
        tag: 'delete',
        srcPos: sourceStart + col,
        destPos: destinationStart + row,
      }
    } else {
      row--
      if (row && recoveryBitSet(vn, stride, offsets, row - 1, col - 1)) {
        dist--
        rangeOps[dist] = {
          tag: 'insert',
          srcPos: sourceStart + col,
          destPos: destinationStart + row,
        }
      } else {
        col--
        if (source[sourceStart + col] !== destination[destinationStart + row]) {
          dist--
          rangeOps[dist] = {
            tag: 'replace',
            srcPos: sourceStart + col,
            destPos: destinationStart + row,
          }
        }
      }
    }
  }
  while (col !== 0) {
    dist--
    col--
    rangeOps[dist] = {
      tag: 'delete',
      srcPos: sourceStart + col,
      destPos: destinationStart + row,
    }
  }
  while (row !== 0) {
    dist--
    row--
    rangeOps[dist] = {
      tag: 'insert',
      srcPos: sourceStart + col,
      destPos: destinationStart + row,
    }
  }
  for (let i = 0; i < rangeOps.length; i++) out.push(rangeOps[i])
}

export function alignHirschberg(
  out: Editop[],
  source: ArrayLike<unknown>,
  sourceStart: number,
  sourceLength: number,
  destination: ArrayLike<unknown>,
  destinationStart: number,
  destinationLength: number,
  maximumDistance: number,
): void {
  const shorter = Math.min(sourceLength, destinationLength)
  let prefix = 0
  while (
    prefix < shorter &&
    source[sourceStart + prefix] === destination[destinationStart + prefix]
  ) {
    prefix++
  }
  let suffix = 0
  while (
    suffix < shorter - prefix &&
    source[sourceStart + sourceLength - suffix - 1] ===
      destination[destinationStart + destinationLength - suffix - 1]
  ) {
    suffix++
  }
  sourceStart += prefix
  destinationStart += prefix
  sourceLength -= prefix + suffix
  destinationLength -= prefix + suffix
  maximumDistance = Math.min(maximumDistance, Math.max(sourceLength, destinationLength))

  const matrixBytes = levenshteinMatrixBytes(
    source,
    sourceStart,
    sourceLength,
    destinationLength,
    maximumDistance,
  )
  if (
    matrixBytes < ALIGNMENT_MATRIX_LIMIT ||
    sourceLength < 65 ||
    destinationLength < 10
  ) {
    recoverMatrixRange(
      out,
      source,
      sourceStart,
      sourceLength,
      destination,
      destinationStart,
      destinationLength,
      maximumDistance,
    )
    return
  }

  const destinationMiddle = Math.floor(destinationLength / 2)
  const rightLength = destinationLength - destinationMiddle
  growHirschbergRows(sourceLength + 1)
  const reversePattern = preparePattern(
    source,
    sourceStart + sourceLength - 1,
    sourceLength,
    -1,
  )
  levenshteinPreparedRow(
    reversePattern,
    destination,
    destinationStart + destinationLength - 1,
    rightLength,
    -1,
    hirschbergRight,
  )
  const forwardPattern = preparePattern(source, sourceStart, sourceLength)
  levenshteinPreparedRow(
    forwardPattern,
    destination,
    destinationStart,
    destinationMiddle,
    1,
    hirschbergLeft,
  )

  let sourceMiddle = 0
  let leftScore = hirschbergLeft[0]
  let rightScore = hirschbergRight[sourceLength]
  let best = leftScore + rightScore
  for (let i = 1; i <= sourceLength; i++) {
    const left = hirschbergLeft[i]
    const right = hirschbergRight[sourceLength - i]
    if (left + right < best) {
      best = left + right
      sourceMiddle = i
      leftScore = left
      rightScore = right
    }
  }

  alignHirschberg(
    out,
    source,
    sourceStart,
    sourceMiddle,
    destination,
    destinationStart,
    destinationMiddle,
    leftScore,
  )
  alignHirschberg(
    out,
    source,
    sourceStart + sourceMiddle,
    sourceLength - sourceMiddle,
    destination,
    destinationStart + destinationMiddle,
    destinationLength - destinationMiddle,
    rightScore,
  )
}
