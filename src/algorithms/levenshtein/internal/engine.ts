import { lcsLengthRange } from '../../lcs/internal/kernel.js'
import { UNBOUNDED_MISSES } from '../../shared/bitmask/blockMasks.js'
import {
  asSequence,
  canonicalRawCutoff,
  conv,
  hasSurrogatePair,
  isNone,
  isSequence,
  normalize,
  normDistCutoff,
  normSimCutoff,
  type MaybeSequence,
  type Sequence,
} from '../../shared/scorerSupport.js'
import type {
  LevenshteinCosts,
  LevenshteinOptions,
  LevenshteinWeights,
} from '../types.js'
import { weightedFloatRow, weightedIntegerRow } from './scratch.js'
import { levenshteinUniform } from './uniform.js'

export { resetWeightedScratch } from './scratch.js'

const INT_ROW_SENTINEL = 0x4000_0000

export const UNIFORM: LevenshteinWeights = [1, 1, 1]

/**
 * Reads the `weights` option, rejecting what no cost can mean.
 *
 * Upstream types these as unsigned integers and its own binding raises on a
 * negative one; this API takes `number`, so a fraction is deliberate but a
 * negative cost, a `NaN` or an infinity is not. None of them has an answer to
 * give: a negative cost makes the cheapest alignment unbounded, and either of
 * the others makes every comparison against them `NaN`. Both were reachable —
 * `NaN` came back as a distance rather than as an error.
 *
 * Every scorer goes through here. The prepared path always did, so a shape the
 * two paths disagreed on used to score as `NaN` directly and throw through
 * `process`.
 */
/** Whether a value carries the three costs by name rather than by position. */
function isLevenshteinCosts(value: object): boolean {
  return 'insertion' in value && 'deletion' in value && 'substitution' in value
}

/**
 * The one reader of the `weights` option, in either spelling.
 *
 * Positional rather than named because this is the scoring path — every
 * weighted comparison runs it, and the kernels destructure three numbers. The
 * named form is one object further out, in {@link levenshteinCosts}, so it is
 * paid for only where it is used: at scorer compilation and in the test harness.
 * Both spellings meet here, so there is one place for them to agree.
 */
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

/** Preserve fractional cutoffs only for the fractional-weight JS extension. */
export function levenshteinRawCutoff(
  cutoff: number | null | undefined,
  integral: boolean,
): number | null {
  if (integral) return canonicalRawCutoff(cutoff)
  if (cutoff == null) return null
  // The same range `canonicalRawCutoff` enforces, minus its truncation: a
  // fractional weighting has fractional distances, so the fraction is the part
  // worth keeping.
  if (!Number.isFinite(cutoff) || cutoff < 0) {
    throw new RangeError('scoreCutoff has to be a finite count of at least 0')
  }
  return cutoff
}

export function rawBound(bound: number, integral: boolean): number {
  return Math.max(0, integral ? Math.floor(bound) : bound)
}

/**
 * {@link parseWeights} with the costs named.
 *
 * Exported because two callers outside scoring need the same answer and must
 * not re-derive it: the flags resolver that tells `process` whether a weighting
 * is symmetric, and the test harness that computes the maximum a normalized
 * score divides by. Always a fresh object, so a caller cannot mutate a shared
 * default and change what a later call reads.
 *
 * Not re-exported from the package barrel — this is shared internals, not API.
 */
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
    const scaledCutoff = Number.isFinite(scoreCutoff)
      ? Math.floor(scoreCutoff / insert)
      : UNBOUNDED_MISSES

    // Factoring equal weights exposes the much faster uniform kernel.
    if (insert === replace) {
      const scaledHint = Number.isFinite(scoreHint)
        ? Math.ceil(scoreHint / insert)
        : UNBOUNDED_MISSES
      return levenshteinUniform(s1, s2, scaledCutoff, scaledHint) * insert
    }

    // Substitution can never win when it costs at least delete + insert.
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

  // How far off the corridor between the two ends the budget can pay to go.
  //
  // Every alignment deletes at least `difference` elements of the longer side,
  // which is `minimum` above and is already known to fit the budget. Beyond
  // that, each insertion has to be matched by another deletion — the two
  // lengths are fixed — so a step off the corridor and back costs
  // `sourceDelete + textInsert`, and the budget buys `excursion` of them.
  //
  // The corridor itself runs from `sourceIndex - difference` to `sourceIndex`,
  // so the reachable columns are that widened by the excursion at both ends.
  // Nothing the row loop touches feeds back into it, so both spans are
  // constants for the whole matrix rather than per-row quantities.
  //
  // The `+ 1` is the same slack the old symmetric radius carried: it costs one
  // diagonal either side and keeps a fractional weighting whose quotient lands
  // a hair below a whole number from banding one diagonal too tight.
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

/**
 * Whether the integer kernel can run without any value leaving `Int32Array`.
 *
 * The all-indel alignment bounds every real cell, and the largest quantity the
 * loop forms is the out-of-band sentinel plus one weight. Requiring both below
 * {@link INT_ROW_SENTINEL} — itself well under `2**31 - 1` — leaves the
 * arithmetic exact and the sentinel unreachable by any genuine cost.
 */
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

/**
 * The generic weighted DP over `Int32Array`.
 *
 * A near-duplicate of {@link weightedFloatDp}, and deliberately so: a single
 * body parameterised by row type is called with two different typed arrays, and
 * the polymorphic element access that produces measured *slower* than the
 * float-only version. Kept apart, each kernel stays monomorphic and the integer
 * one runs about 1.5x the float one. Any change here belongs in both.
 */
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
    // The diagonal predecessor of the row's first cell, read before column zero
    // is overwritten. Once the band has left column one that is the previous
    // row's own first cell, which the band's one-column-a-row drift leaves
    // exactly here; the two coincide only while `low` is one.
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

/** The generic weighted DP over `Float64Array`. See {@link weightedIntegerDp}. */
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
    // See {@link weightedIntegerDp} for why the diagonal is read from `low - 1`.
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

/**
 * Weighted Levenshtein distance.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
export function levenshteinDistanceImpl(
  s1: Sequence,
  s2: Sequence,
  options?: LevenshteinOptions,
): number {
  // `levenshteinDistance(a, b)` on two BMP strings is the call shape almost
  // every caller writes, and the generic path below spends more on describing
  // it than the kernel spends scoring an eight-character pair: an options
  // object to allocate and read four properties off, a weights triple to
  // validate, `conv`'s two-element tuple to allocate and destructure, and a
  // cutoff to convert on the way in and collapse on the way out — none of which
  // this shape can affect.
  //
  // What survives is the surrogate test, which is what decides the pair can be
  // scored as strings at all; it is the same test and in the same order as
  // `convPair`, so a pair takes the same representation either way.
  //
  // Uniform weights and no cutoff reduce `distance_` to exactly this call: the
  // empty-input returns are `levenshteinUniform`'s own, the zero-cost and
  // Indel-equivalent branches cannot be reached at `[1, 1, 1]`, the minimum-
  // distance rejection cannot fire without a cutoff, and the `* insert` scaling
  // is a multiplication by one. With no cutoff the final clamp returns its input.
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
  const [a, b] = conv(s1, s2, options?.processor)
  const cutoff = levenshteinRawCutoff(options?.scoreCutoff, integral)
  const bound = cutoff ?? Number.MAX_SAFE_INTEGER
  const hint = options?.scoreHint ?? bound
  const distance = distance_(a, b, weights, bound, hint)
  return cutoff === null || distance <= cutoff ? distance : cutoff + 1
}

/**
 * Levenshtein similarity: the maximum possible distance minus the actual one.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
export function levenshteinSimilarityImpl(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinOptions = {},
): number {
  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = conv(s1, s2, options.processor)
  const max = maximum(a, b, weights)
  const cutoff = levenshteinRawCutoff(options.scoreCutoff, integral)
  const bound =
    cutoff == null ? Number.MAX_SAFE_INTEGER : rawBound(max - cutoff, integral)
  const hint =
    options.scoreHint == null ? bound : rawBound(max - options.scoreHint, integral)
  const similarity = max - distance_(a, b, weights, bound, hint)
  return cutoff === null || similarity >= cutoff ? similarity : 0
}

/**
 * {@link levenshteinDistance} normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
export function levenshteinNormalizedDistanceImpl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: LevenshteinOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b, weights)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : rawBound(options.scoreCutoff * max, integral)
  const hint =
    options.scoreHint == null ? cutoff : rawBound(options.scoreHint * max, integral)
  const norm = normalize(distance_(a, b, weights, cutoff, hint), max)
  return normDistCutoff(norm, options.scoreCutoff)
}

/**
 * {@link levenshteinSimilarity} normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
export function levenshteinNormalizedSimilarityImpl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: LevenshteinOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const weights = parseWeights(options.weights)
  const integral = integralWeights(weights)
  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b, weights)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : rawBound((1 - options.scoreCutoff) * max, integral)
  const hint =
    options.scoreHint == null ? cutoff : rawBound((1 - options.scoreHint) * max, integral)
  const norm = normalize(distance_(a, b, weights, cutoff, hint), max)
  return normSimCutoff(1 - norm, options.scoreCutoff)
}
