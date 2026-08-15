import { checkedStartGeneration } from '../algorithms/bitmask/blockMasks.js'
import type { PatternMask } from '../algorithms/bitmask/pattern.js'
import {
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  lcsSeqLengthRange,
  prepareLcsPattern,
} from '../algorithms/lcs/implementation.js'
import {
  validateSequence,
  convPair,
  isMissing,
} from '../algorithms/shared/scorerSupport.js'
import type { FuzzInput, FuzzOptions, ScoreAlignment } from './types.js'

function indelNormSim(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return indelNormSimRange(a, 0, a.length, b, 0, b.length, scoreCutoff)
}

function indelNormSimRange(
  a: ArrayLike<unknown>,
  start1: number,
  len1: number,
  b: ArrayLike<unknown>,
  start2: number,
  len2: number,
  scoreCutoff: number,
): number {
  const maximum = len1 + len2
  if (maximum === 0) return 1 >= scoreCutoff ? 1 : 0

  const ceiling = 1 - Math.abs(len1 - len2) / maximum
  if (ceiling < scoreCutoff) return 0

  const budget =
    scoreCutoff > 0 ? Math.floor(maximum * (1 - scoreCutoff)) + 1 : maximum + 1
  const lcs = lcsSeqLengthRange(a, start1, len1, b, start2, len2, budget)

  const sim = 1 - (maximum - 2 * lcs) / maximum
  return sim >= scoreCutoff ? sim : 0
}

export function indelNormSimHeld(
  pattern: PatternMask,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
  scoreCutoff: number,
): number {
  const maximum = patternLength + textLength
  if (maximum === 0) return 1 >= scoreCutoff ? 1 : 0

  const ceiling = 1 - Math.abs(patternLength - textLength) / maximum
  if (ceiling < scoreCutoff) return 0

  const lcs = lcsSeqLengthPrepared(pattern, text, textStart, textLength)
  const sim = 1 - (maximum - 2 * lcs) / maximum

  return sim >= scoreCutoff ? sim : 0
}

export function ratioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return indelNormSim(a, b, scoreCutoff / 100) * 100
}

/**
 * The cutoff is divided by 100 on the way in and the score multiplied by 100 on
 * the way out, matching `ratioConverted`. Comparing percentages instead is
 * algebraically the same test and a different one in floating point, and the
 * two paths then disagree by a ULP on scores fed back as cutoffs.
 */
export function ratioHeld(
  pattern: PatternMask,
  patternLength: number,
  text: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  if (scoreCutoff > 100) return 0

  const cutoff = scoreCutoff / 100
  const textLength = text.length
  const maximum = patternLength + textLength
  if (maximum === 0) return 100

  const ceiling = 1 - Math.abs(patternLength - textLength) / maximum
  if (ceiling < cutoff) return 0

  const required = Math.max(0, Math.floor((cutoff * maximum) / 2))
  const lcs =
    cutoff >= 0.7 && maximum >= 128
      ? lcsSeqLengthPreparedBounded(pattern, text, 0, textLength, required)
      : lcsSeqLengthPrepared(pattern, text, 0, textLength)
  if (lcs < 0) return 0

  const sim = 1 - (maximum - 2 * lcs) / maximum
  return sim >= cutoff ? sim * 100 : 0
}

export function ratio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return ratioConverted(a, b, options.scoreCutoff ?? 0)
}

export interface CharSet {
  readonly direct: Uint8Array
  readonly wide: ReadonlySet<unknown> | null
}

const HIGH_TABLE_LIMIT = 2048

/**
 * Build the pruning set for `s`, split out so a caller scoring one query
 * against many candidates builds it once.
 *
 * `NaN` is kept deliberately: it lands in {@link CharSet.wide}, where `Set`
 * matches it against itself although the kernels refuse to, so its window is
 * scanned and scores nothing. Dropping it would prune that window instead — a
 * stronger filter than this is documented to be, which could move an alignment.
 */
export function charSetOf(s: ArrayLike<unknown>): CharSet {
  let wide: Set<unknown> | null = null

  if (typeof s === 'string') {
    const direct = new Uint8Array(256)
    let codes: Set<number> | null = null
    let highest = -1
    for (let i = 0; i < s.length; i++) {
      const code = s.charCodeAt(i)
      if (code < 256) direct[code] = 1
      else {
        ;(codes ??= new Set<number>()).add(code)
        if (code > highest) highest = code
      }
    }

    if (codes === null || highest >= HIGH_TABLE_LIMIT) return { direct, wide: codes }

    const wideDirect = new Uint8Array(highest + 1)
    wideDirect.set(direct)
    for (const code of codes) wideDirect[code] = 1
    return { direct: wideDirect, wide: null }
  }

  let direct: Uint8Array | null = null
  for (let i = 0; i < s.length; i++) {
    const value = s[i]
    if (typeof value === 'number' && value >= 0 && value < 256 && (value | 0) === value) {
      ;(direct ??= new Uint8Array(256))[value] = 1
    } else {
      ;(wide ??= new Set<unknown>()).add(value)
    }
  }

  return { direct: direct ?? emptyTable(), wide }
}

let shared: Uint8Array | null = null
function emptyTable(): Uint8Array {
  return (shared ??= new Uint8Array(256))
}

const BISECTION_SCRATCH_LIMIT = 1 << 16
let bisectionScores: Uint32Array | null = null
let bisectionStamps: Int32Array | null = null
let bisectionWindows: number[] = []
let bisectionNextWindows: number[] = []
let bisectionGeneration = 0

const BISECTION_GENERATION_LIMIT = 0x7fff_ffff

function nextBisectionGeneration(): number {
  bisectionGeneration++
  if (bisectionGeneration >= BISECTION_GENERATION_LIMIT) {
    bisectionStamps = null
    bisectionGeneration = 1
  }

  return bisectionGeneration
}

function bisectionScoresFor(size: number): Uint32Array {
  const held = bisectionScores
  if (held !== null && held.length >= size) return held

  let next = 256
  while (next < size) next *= 2
  bisectionScores = new Uint32Array(next)
  return bisectionScores
}

function bisectionStampsFor(size: number): Int32Array {
  const held = bisectionStamps
  if (held !== null && held.length >= size) return held

  let next = 256
  while (next < size) next *= 2
  bisectionStamps = new Int32Array(next)
  return bisectionStamps
}

export function resetPartialRatioScratch(startGeneration = 0): void {
  bisectionScores = null
  bisectionStamps = null
  bisectionWindows = []
  bisectionNextWindows = []
  bisectionGeneration = checkedStartGeneration(startGeneration)
}

/**
 * Shortest window of `s2` a needle of `len1` could still score at `cutoff`.
 * Deliberately floored: it may answer below the true minimum, so a scan can
 * visit a window the kernel rejects but can never skip one it would have scored.
 */
function minimumWindow(len1: number, cutoff: number): number {
  const estimate = Math.floor((cutoff * len1) / (2 - cutoff))
  return estimate < 1 ? 1 : estimate
}

export function partialRatioImpl(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number,
  prepared: PatternMask = prepareLcsPattern(s1, 0, s1.length),
  scoreOnly = false,
  preparedCharSet?: CharSet,
): ScoreAlignment {
  return partialRatioScan(s1, s2, scoreCutoff, prepared, scoreOnly, preparedCharSet)
}

/**
 * Both inputs must already share an element representation — the pruning set
 * compares them with `===`. `preparedCharSet`, when given, must be
 * {@link charSetOf} over this `s1` in the representation it arrived in.
 */
function partialRatioScan(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff: number,
  pattern: PatternMask,
  scoreOnly: boolean,
  preparedCharSet?: CharSet,
): ScoreAlignment {
  const len1 = s1.length
  const len2 = s2.length

  if (scoreCutoff > 1) {
    return { score: 0, srcStart: 0, srcEnd: len1, destStart: 0, destEnd: len1 }
  }

  if (len1 === 0) {
    return { score: 0, srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 }
  }

  const charSet = preparedCharSet ?? charSetOf(s1)
  const direct = charSet.direct
  const narrow = direct.length
  const wide = charSet.wide

  const text = typeof s2 === 'string' ? s2 : null

  const holds = (index: number): boolean => {
    if (text === null) {
      const value = s2[index]
      if (
        typeof value === 'number' &&
        value >= 0 &&
        value < 256 &&
        (value | 0) === value
      ) {
        return direct[value] !== 0
      }
      return wide !== null && wide.has(value)
    }

    const code = text.charCodeAt(index)
    return code < narrow ? direct[code] !== 0 : wide !== null && wide.has(code)
  }

  const res = {
    score: 0,
    srcStart: 0,
    srcEnd: len1,
    destStart: 0,
    destEnd: len1,
  }
  let cutoff = scoreCutoff

  let minWindow = minimumWindow(len1, cutoff)

  const acceptKnownScore = (score: number, start: number, end: number): boolean => {
    if (score <= res.score) return false

    res.score = score
    cutoff = score
    minWindow = minimumWindow(len1, score)
    res.destStart = start
    res.destEnd = end

    if (res.score === 1) {
      res.score = 100
      return true
    }

    return false
  }

  const consider = (start: number, end: number): boolean => {
    const lsRatio = indelNormSimHeld(pattern, len1, s2, start, end - start, cutoff)
    return acceptKnownScore(lsRatio, start, end)
  }

  const windowDistance = (start: number): number => {
    const lcs = lcsSeqLengthPrepared(pattern, s2, start, len1)
    return 2 * (len1 - lcs)
  }

  const scanPrefix = (): void => {
    for (let i = minWindow; i < len1; i++) {
      if (!holds(i - 1)) continue
      consider(0, i)
    }
  }

  const scanSuffix = (): boolean => {
    for (let i = len2 - len1; i < len2; i++) {
      if (len2 - i < minWindow) break
      if (!holds(i)) continue
      if (consider(i, len2)) return true
    }
    return false
  }

  const scanInterior = (): boolean => {
    const lastInterior = len2 - len1 - 1
    if (lastInterior >= 64) {
      const windowCount = lastInterior + 1
      const held = windowCount <= BISECTION_SCRATCH_LIMIT
      const generation = nextBisectionGeneration()
      const scores = held ? bisectionScoresFor(windowCount) : new Uint32Array(windowCount)
      const stamps = held ? bisectionStampsFor(windowCount) : new Int32Array(windowCount)
      let distanceToBeat =
        Math.floor(2 * len1 * (1 - cutoff) + Number.EPSILON * 2 * len1) + 1
      let windows = held ? bisectionWindows : []
      let nextWindows = held ? bisectionNextWindows : []
      windows.length = 0
      nextWindows.length = 0
      windows.push(0, lastInterior)

      while (windows.length !== 0) {
        for (let window = 0; window < windows.length; window += 2) {
          const first = windows[window]
          const last = windows[window + 1]

          if (stamps[first] !== generation) {
            const distance = windowDistance(first)
            scores[first] = distance
            stamps[first] = generation
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (acceptKnownScore(score, first, first + len1)) return true
            }
          }
          if (stamps[last] !== generation) {
            const distance = windowDistance(last)
            scores[last] = distance
            stamps[last] = generation
            const score = 1 - distance / (2 * len1)
            if (distance < distanceToBeat && score >= cutoff) {
              distanceToBeat = distance
              if (acceptKnownScore(score, last, last + len1)) return true
            }
          }

          const cellDiff = last - first
          if (cellDiff <= 1) continue

          const knownEdits = Math.abs(scores[first] - scores[last])
          const maxImprovement =
            Math.floor((cellDiff - Math.floor(knownEdits / 2)) / 2) * 2
          const minimumPossible = Math.min(scores[first], scores[last]) - maxImprovement
          if (minimumPossible < distanceToBeat) {
            const center = first + Math.floor(cellDiff / 2)
            nextWindows.push(first, center, center, last)
          }
        }

        const oldWindows = windows
        windows = nextWindows
        nextWindows = oldWindows
        nextWindows.length = 0
      }
    } else {
      for (let i = 0; i < len2 - len1; i++) {
        if (!holds(i + len1 - 1)) continue
        if (consider(i, i + len1)) return true
      }
    }
    return false
  }

  if (scoreOnly) {
    if (scanInterior() || scanSuffix()) return res
    scanPrefix()
  } else {
    scanPrefix()
    if (scanInterior() || scanSuffix()) return res
  }

  res.score *= 100
  return res
}

function swapAlignment(a: ScoreAlignment): ScoreAlignment {
  return {
    score: a.score,
    srcStart: a.destStart,
    srcEnd: a.destEnd,
    destStart: a.srcStart,
    destEnd: a.srcEnd,
  }
}

export function partialRatioAlignment_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): ScoreAlignment | null {
  if (isMissing(s1) || isMissing(s2)) return null

  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))

  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0)
}

export function partialAlignmentConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  cutoff: number,
  scoreOnly = false,
  preparedA?: PatternMask,
  preparedCharSetA?: CharSet,
): ScoreAlignment | null {
  if (cutoff > 100) return null

  let scoreCutoff = cutoff

  if (a.length === 0 && b.length === 0) {
    return { score: 100, srcStart: 0, srcEnd: 0, destStart: 0, destEnd: 0 }
  }

  const s1Shorter = a.length <= b.length
  const shorter = s1Shorter ? a : b
  const longer = s1Shorter ? b : a

  const aIsNeedle = shorter === a
  let res = partialRatioImpl(
    shorter,
    longer,
    scoreCutoff / 100,
    aIsNeedle ? preparedA : undefined,
    scoreOnly,
    aIsNeedle ? preparedCharSetA : undefined,
  )

  if (res.score !== 100 && a.length === b.length) {
    scoreCutoff = Math.max(scoreCutoff, res.score)
    const res2 = partialRatioImpl(
      longer,
      shorter,
      scoreCutoff / 100,
      undefined,
      scoreOnly,
    )
    if (res2.score > res.score) res = swapAlignment(res2)
  }

  if (res.score < scoreCutoff) return null

  return s1Shorter ? res : swapAlignment(res)
}

export function partialRatioConverted(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  return partialAlignmentConverted(a, b, scoreCutoff, true)?.score ?? 0
}

export function partialRatio_impl(
  s1: FuzzInput,
  s2: FuzzInput,
  options: FuzzOptions = {},
): number {
  if (isMissing(s1) || isMissing(s2)) return 0

  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return partialAlignmentConverted(a, b, options.scoreCutoff ?? 0, true)?.score ?? 0
}
