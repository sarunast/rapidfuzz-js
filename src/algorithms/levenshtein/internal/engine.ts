import {
  canonicalRawCutoff,
  canonicalSimilarityCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
} from '../../../core/scoring/builtIn/cutoff.js'
import { convPair, hasSurrogatePair, isSequence } from '../../../core/sequence.js'
import type { Sequence } from '../../../core/types.js'
import { lcsLengthRange } from '../../lcs/internal/kernel.js'
import type {
  LevenshteinCosts,
  LevenshteinOptions,
  LevenshteinWeights,
} from '../types.js'
import { weightedFloatRow, weightedIntegerRow } from './scratch.js'
import { levenshteinUniform } from './uniform.js'

export { resetWeightedScratch } from './scratch.js'

const INT_ROW_SENTINEL = 0x4000_0000

const UNIFORM: LevenshteinWeights = [1, 1, 1]

function isLevenshteinCosts(value: object): boolean {
  return 'insertion' in value && 'deletion' in value && 'substitution' in value
}

export function parseWeights(value: unknown): LevenshteinWeights {
  if (value == null) return UNIFORM

  let insertion: unknown
  let deletion: unknown
  let substitution: unknown

  if (typeof value === 'object' && isLevenshteinCosts(value)) {
    insertion = Reflect.get(value, 'insertion')
    deletion = Reflect.get(value, 'deletion')
    substitution = Reflect.get(value, 'substitution')
  } else if (isSequence(value) && value.length === 3) {
    insertion = value[0]
    deletion = value[1]
    substitution = value[2]
  } else {
    throw new TypeError('weights must contain insertion, deletion and replacement costs')
  }

  if (
    typeof insertion !== 'number' ||
    typeof deletion !== 'number' ||
    typeof substitution !== 'number'
  ) {
    throw new TypeError('weights must contain numbers')
  }
  if (
    !(insertion >= 0 && insertion !== Infinity) ||
    !(deletion >= 0 && deletion !== Infinity) ||
    !(substitution >= 0 && substitution !== Infinity)
  ) {
    throw new TypeError('weights must contain finite costs of at least zero')
  }

  return [insertion, deletion, substitution]
}

export function integralWeights(weights: LevenshteinWeights): boolean {
  return weights.every(Number.isInteger)
}

export function levenshteinRawCutoff(
  cutoff: number | null | undefined,
  integral: boolean,
): number | null {
  if (integral) return canonicalRawCutoff(cutoff)
  if (cutoff == null) return null
  if (!Number.isFinite(cutoff) || cutoff < 0) {
    throw new RangeError('scoreCutoff has to be a finite count of at least 0')
  }
  return cutoff
}

export function levenshteinSimilarityCutoff(
  cutoff: number | null | undefined,
  integral: boolean,
): number | null {
  if (integral) return canonicalSimilarityCutoff(cutoff)
  return levenshteinRawCutoff(cutoff, false)
}

export function rawBound(bound: number, integral: boolean): number {
  return Math.max(0, integral ? Math.floor(bound) : bound)
}

export function levenshteinCosts(value: unknown): LevenshteinCosts {
  const [insertion, deletion, substitution] = parseWeights(value)
  return { insertion, deletion, substitution }
}

export function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  weights: LevenshteinWeights,
  scoreCutoff: number = Number.MAX_SAFE_INTEGER,
  scoreHint: number = scoreCutoff,
): number {
  const [insert, delete_, replace] = weights
  const len1 = s1.length
  const len2 = s2.length

  if (len1 === 0) return len2 * insert
  if (len2 === 0) return len1 * delete_

  if (insert === 0 && delete_ === 0) return 0

  if (insert === delete_ && insert > 0) {
    const scaledCutoff = Math.floor(scoreCutoff / insert)

    if (insert === replace) {
      const scaledHint = Math.ceil(scoreHint / insert)
      return levenshteinUniform(s1, s2, scaledCutoff, scaledHint) * insert
    }

    if (replace >= insert + delete_) {
      const lcs = lcsLengthRange(s1, 0, len1, s2, 0, len2, scaledCutoff)
      return (len1 + len2 - 2 * lcs) * insert
    }
  }

  const minimum = len1 >= len2 ? (len1 - len2) * delete_ : (len2 - len1) * insert
  if (minimum > scoreCutoff) return scoreCutoff + 1

  const shorter = Math.min(len1, len2)
  let prefix = 0
  while (prefix < shorter && s1[prefix] === s2[prefix]) prefix++
  let suffix = 0
  while (suffix < shorter - prefix && s1[len1 - suffix - 1] === s2[len2 - suffix - 1]) {
    suffix++
  }

  const trimmed1 = len1 - prefix - suffix
  const trimmed2 = len2 - prefix - suffix
  if (trimmed1 === 0) return trimmed2 * insert
  if (trimmed2 === 0) return trimmed1 * delete_

  let source = s1
  let text = s2
  let sourceLength = trimmed1
  let textLength = trimmed2
  let sourceDelete = delete_
  let textInsert = insert
  if (textLength > sourceLength) {
    source = s2
    text = s1
    sourceLength = trimmed2
    textLength = trimmed1
    sourceDelete = insert
    textInsert = delete_
  }

  const bounded = Number.isFinite(scoreCutoff) && scoreCutoff < Number.MAX_SAFE_INTEGER
  const step = sourceDelete + textInsert

  const difference = sourceLength - textLength
  const excursion =
    bounded && step > 0
      ? Math.floor((scoreCutoff - difference * sourceDelete) / step) + 1
      : Math.max(sourceLength, textLength)
  const belowSpan = bounded && step > 0 ? difference + excursion : excursion

  return integerRowFits(sourceLength, textLength, sourceDelete, textInsert, replace)
    ? weightedIntegerDp(
        source,
        text,
        prefix,
        sourceLength,
        textLength,
        sourceDelete,
        textInsert,
        replace,
        belowSpan,
        excursion,
      )
    : weightedFloatDp(
        source,
        text,
        prefix,
        sourceLength,
        textLength,
        sourceDelete,
        textInsert,
        replace,
        belowSpan,
        excursion,
      )
}

function integerRowFits(
  sourceLength: number,
  textLength: number,
  sourceDelete: number,
  textInsert: number,
  replace: number,
): boolean {
  if (
    !Number.isSafeInteger(sourceDelete) ||
    !Number.isSafeInteger(textInsert) ||
    !Number.isSafeInteger(replace) ||
    sourceDelete < 0 ||
    textInsert < 0 ||
    replace < 0
  ) {
    return false
  }

  const largestCost = sourceLength * sourceDelete + textLength * textInsert
  const largestWeight = Math.max(sourceDelete, textInsert, replace)
  return largestCost + largestWeight < INT_ROW_SENTINEL
}

function weightedIntegerDp(
  source: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  prefix: number,
  sourceLength: number,
  textLength: number,
  sourceDelete: number,
  textInsert: number,
  replace: number,
  belowSpan: number,
  aboveSpan: number,
): number {
  const row = weightedIntegerRow(textLength + 1)
  for (let j = 0; j <= textLength; j++) row[j] = j * textInsert

  for (let i = 1; i <= sourceLength; i++) {
    const low = Math.max(1, i - belowSpan)
    const high = Math.min(textLength, i + aboveSpan)
    let prevDiag = row[low - 1]
    row[0] = i * sourceDelete
    const a = source[prefix + i - 1]

    if (low > 1) row[low - 1] = INT_ROW_SENTINEL

    for (let j = low; j <= high; j++) {
      const above = row[j]
      const cost = Math.min(
        above + sourceDelete,
        row[j - 1] + textInsert,
        prevDiag + (a === text[prefix + j - 1] ? 0 : replace),
      )
      prevDiag = above
      row[j] = cost
    }
    if (high < textLength) row[high + 1] = INT_ROW_SENTINEL
  }

  return row[textLength]
}

function weightedFloatDp(
  source: ArrayLike<unknown>,
  text: ArrayLike<unknown>,
  prefix: number,
  sourceLength: number,
  textLength: number,
  sourceDelete: number,
  textInsert: number,
  replace: number,
  belowSpan: number,
  aboveSpan: number,
): number {
  const row = weightedFloatRow(textLength + 1)
  for (let j = 0; j <= textLength; j++) row[j] = j * textInsert

  for (let i = 1; i <= sourceLength; i++) {
    const low = Math.max(1, i - belowSpan)
    const high = Math.min(textLength, i + aboveSpan)
    let prevDiag = row[low - 1]
    row[0] = i * sourceDelete
    const a = source[prefix + i - 1]

    if (low > 1) row[low - 1] = Infinity

    for (let j = low; j <= high; j++) {
      const above = row[j]
      const cost = Math.min(
        above + sourceDelete,
        row[j - 1] + textInsert,
        prevDiag + (a === text[prefix + j - 1] ? 0 : replace),
      )
      prevDiag = above
      row[j] = cost
    }
    if (high < textLength) row[high + 1] = Infinity
  }

  return row[textLength]
}

export function maximum(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  weights: LevenshteinWeights,
): number {
  const [insert, delete_, replace] = weights
  const len1 = s1.length
  const len2 = s2.length

  const indel = len1 * delete_ + len2 * insert

  return len1 >= len2
    ? Math.min(indel, len2 * replace + (len1 - len2) * delete_)
    : Math.min(indel, len1 * replace + (len2 - len1) * insert)
}

export function levenshteinDistanceImpl(
  s1: Sequence,
  s2: Sequence,
  options?: LevenshteinOptions,
): number {
  if (
    options === undefined &&
    typeof s1 === 'string' &&
    typeof s2 === 'string' &&
    !hasSurrogatePair(s1) &&
    !hasSurrogatePair(s2)
  ) {
    return levenshteinUniform(s1, s2, Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER)
  }

  const weights = parseWeights(options?.weights)
  const integral = integralWeights(weights)
  const [a, b] = convPair(s1, s2)
  const cutoff = levenshteinRawCutoff(options?.scoreCutoff, integral)
  const bound = cutoff ?? Number.MAX_SAFE_INTEGER
  const hint = options?.scoreHint ?? bound
  const distance = distance_(a, b, weights, bound, hint)
  return cutoff === null || distance <= cutoff ? distance : cutoff + 1
}

export function levenshteinSimilarityImpl(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinOptions = {},
): number {
  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b, weights)
  const cutoff = levenshteinSimilarityCutoff(options.scoreCutoff, integral)
  const bound =
    cutoff === null ? Number.MAX_SAFE_INTEGER : rawBound(max - cutoff, integral)
  const hint =
    options.scoreHint == null ? bound : rawBound(max - options.scoreHint, integral)
  return simCutoff(max - distance_(a, b, weights, bound, hint), cutoff)
}

export function levenshteinNormalizedDistanceImpl(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinOptions = {},
): number {
  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b, weights)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : rawBound(options.scoreCutoff * max, integral)
  const hint =
    options.scoreHint == null ? cutoff : rawBound(options.scoreHint * max, integral)
  return normDistCutoff(
    normalizeDistance(distance_(a, b, weights, cutoff, hint), max),
    options.scoreCutoff,
  )
}

export function levenshteinNormalizedSimilarityImpl(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinOptions = {},
): number {
  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b, weights)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : rawBound((1 - options.scoreCutoff) * max, integral)
  const hint =
    options.scoreHint == null ? cutoff : rawBound((1 - options.scoreHint) * max, integral)
  const norm = normalizeDistance(distance_(a, b, weights, cutoff, hint), max)
  return normSimCutoff(1 - norm, options.scoreCutoff)
}
