import {
  alignRepresentation,
  canonicalRawCutoff,
  conv,
  hasSurrogatePair,
  normalize,
  normDistCutoff,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MaybeSequence,
  isNone,
  asSequence,
  type EditopsOptions,
  isSequence,
  withChoicePreparer,
  prepareScorerChoice,
  preparedScorerSequence,
  scorerSequence,
  sharesWideAffix,
  type PrepareScorer,
  type PreparedScore,
  withPreparedFlags,
  type ConfigureOptions,
  type ConfiguredFlags,
  type NormalizedScorer,
  type Scorer,
  type ScorerFlags,
} from '../_common.js'
import {
  levenshteinMatrix,
  levenshteinMatrixBytes,
  rowBitSet,
  shiftedRowBitSet,
} from './_bitParallel.js'
import {
  lcsLengthRange,
  levenshteinPrepared,
  levenshteinPreparedRow,
  levenshteinSmallBand,
  levenshteinUniform,
  preparePattern,
  UNBOUNDED_MISSES,
} from './_bitVector/index.js'
import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from './editops.js'

const ALIGNMENT_MATRIX_LIMIT = 1024 * 1024
let hirschbergLeft = new Uint32Array(0)
let hirschbergRight = new Uint32Array(0)

/**
 * Out-of-band marker for the integer weighted DP, standing in for the float
 * kernel's `Infinity`. `2**30` is above any cost {@link integerRowFits} admits
 * and leaves room to add a weight without leaving `Int32Array`.
 */
const INT_ROW_SENTINEL = 0x4000_0000

/**
 * Scratch rows for the generic weighted DP, retained the way the bit-parallel
 * kernels retain theirs.
 *
 * The DP overwrites positions `0..textLength` before reading any of them, so
 * carrying values over from an earlier call cannot be observed. Non-uniform
 * weights are the one scoring path left that allocated per comparison, which is
 * exactly the shape `process.extract` repeats thousands of times.
 */
let weightedFloatScratch: Float64Array | null = null
let weightedIntegerScratch: Int32Array | null = null

/**
 * Drop the retained weighted rows and Hirschberg halves. Benchmark-only — see
 * `resetSharedScratch` in `_bitVector/index.ts` for why the benchmarks need it.
 */
export function resetWeightedScratch(): void {
  weightedFloatScratch = null
  weightedIntegerScratch = null
  hirschbergLeft = new Uint32Array(0)
  hirschbergRight = new Uint32Array(0)
}

function weightedFloatRow(needed: number): Float64Array {
  const held = weightedFloatScratch
  if (held !== null && held.length >= needed) return held
  let size = held === null ? 64 : held.length
  while (size < needed) size *= 2
  weightedFloatScratch = new Float64Array(size)
  return weightedFloatScratch
}

function weightedIntegerRow(needed: number): Int32Array {
  const held = weightedIntegerScratch
  if (held !== null && held.length >= needed) return held
  let size = held === null ? 64 : held.length
  while (size < needed) size *= 2
  weightedIntegerScratch = new Int32Array(size)
  return weightedIntegerScratch
}

function growHirschbergRows(needed: number): void {
  if (hirschbergLeft.length >= needed) return
  let size = Math.max(128, hirschbergLeft.length)
  while (size < needed) size *= 2
  hirschbergLeft = new Uint32Array(size)
  hirschbergRight = new Uint32Array(size)
}

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

function alignHirschberg(
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

/** Cost of an insertion, a deletion, and a substitution, in that order. */
export type LevenshteinWeights = readonly [
  insert: number,
  delete_: number,
  replace: number,
]

/**
 * The same three costs, named.
 *
 * Upstream takes only the tuple, because that is what its C++ signature takes.
 * Nothing at a call site says which of `[1, 1, 2]` is the substitution, and
 * getting the order wrong produces a plausible wrong number rather than an
 * error — so the named form is the one to reach for, and the tuple stays for
 * parity with upstream's docs.
 */
export interface LevenshteinCosts {
  readonly insertion: number
  readonly deletion: number
  readonly substitution: number
}

/** Either spelling of the three costs. */
export type LevenshteinWeightsInput = LevenshteinWeights | LevenshteinCosts

export interface LevenshteinOptions extends ScorerOptions {
  /**
   * Defaults to uniform costs of `1`. Accepts
   * `{ insertion, deletion, substitution }` or the positional
   * `[insertion, deletion, substitution]`.
   */
  weights?: LevenshteinWeightsInput | undefined
}

const UNIFORM: LevenshteinWeights = [1, 1, 1]

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
 * paid for only where it is used: at configure time, and in the test harness.
 * Both spellings meet here, so there is one place for them to agree.
 */
function parseWeights(value: unknown): LevenshteinWeights {
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

function integralWeights(weights: LevenshteinWeights): boolean {
  return weights.every(Number.isInteger)
}

/** Preserve fractional cutoffs only for the fractional-weight JS extension. */
function levenshteinRawCutoff(
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

function rawBound(bound: number, integral: boolean): number {
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

function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  weights: LevenshteinWeights,
  scoreCutoff = Number.MAX_SAFE_INTEGER,
  scoreHint = scoreCutoff,
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

function maximum(
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
function levenshteinDistance_impl(
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
function levenshteinSimilarity_impl(
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
function levenshteinNormalizedDistance_impl(
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
function levenshteinNormalizedSimilarity_impl(
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

/**
 * Options for {@link levenshteinEditops}, which alone among the metrics that
 * expose edit operations takes a hint — the others have no band to narrow.
 */
export interface LevenshteinEditopsOptions extends EditopsOptions {
  /**
   * Estimate of the distance, which buys a narrower alignment band.
   *
   * It cannot change the *length* of the edit script. It can change which
   * optimal script comes back, but only for inputs large enough to need the
   * recursive path: a hint that pays for itself replaces the assumed distance
   * with the real one, that shrinks the matrix the dispatch is sizing, and a
   * pair that then fits takes the exact matrix instead of being split. Both
   * answers are optimal; only one of them is upstream's.
   */
  scoreHint?: number | undefined
}

/**
 * Edit operations that turn `s1` into `s2`.
 *
 * The alignment follows Hyyrö's bit-parallel algorithm, and over a matrix it
 * recovers the operations upstream produces, one for one. Uniform weights only,
 * as upstream.
 *
 * Upstream always builds that matrix, at a bit per cell. This does too while it
 * fits in {@link ALIGNMENT_MATRIX_LIMIT}, and splits the alignment in half
 * recursively when it does not — an alignment of two 16k inputs is 33MB of
 * matrix. Past that point the operations are still an optimal edit script of
 * the same length, but not always upstream's choice among the optimal ones.
 *
 * That is not a split rule waiting to be fixed. Upstream's recovery walks the
 * whole matrix backwards, preferring a deletion at each step, and which script
 * that yields is a property of the whole matrix rather than of any one cell the
 * path passes through. Splitting at exactly the column where upstream's path
 * crosses the middle row is not enough: measured over large random pairs, of
 * the cases that disagree, more than a third split on upstream's own path and
 * still recovered a different script, because each half is then recovered from
 * its own matrix, whose row differences are not the ones upstream read.
 */
export function levenshteinEditops(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinEditopsOptions = {},
): Editops {
  const [full1, full2] = conv(s1, s2, options.processor)
  const ops: Editop[] = []
  let maximum = Math.max(full1.length, full2.length)

  // A hint buys a narrower alignment band, but only by finding the distance
  // first — so the alignment is computed twice over. Upstream takes that trade
  // only when the hint promises to more than halve the second pass, and this
  // follows it: without a hint nothing extra runs.
  const hint =
    options.scoreHint == null ? null : Math.max(31, Math.floor(options.scoreHint))
  if (hint !== null && 2 * hint < maximum) {
    maximum = distance_(full1, full2, UNIFORM, maximum, hint)
  }

  alignHirschberg(ops, full1, 0, full1.length, full2, 0, full2.length, maximum)
  return editopsFromValidated(ops, full1.length, full2.length)
}

/** {@link levenshteinEditops} expressed as blocks. */
export function levenshteinOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: LevenshteinEditopsOptions = {},
): Opcodes {
  return levenshteinEditops(s1, s2, options).toOpcodes()
}

/**
 * Widest budget {@link levenshteinSmallBand} can band inside one word.
 *
 * Its diagonal band is `2 * budget + 1` wide and it holds the band in a single
 * 32-bit word, so anything past 15 would have the part that does not fit
 * dropped and come back too large.
 */
const MAX_BAND_BUDGET = 15

type PreparedLevenshteinKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

/**
 * Whether the held-pattern kernel beats rebuilding the query's masks per choice.
 *
 * Four things have to hold, and each of them is what makes the prepared result
 * identical to the unprepared one rather than merely close.
 *
 * The weights must be uniform, since only then does {@link distance_} reach the
 * plain Myers kernel rather than the LCS or generic ones.
 *
 * The budget and the hint must both be wide enough that {@link levenshteinUniform}
 * would have run its unbounded kernel. A banded run reports `budget + 1` for
 * anything out of reach, which the exact kernel below never does; narrower
 * budgets are served by {@link preparedBandWorthwhile} instead.
 *
 * The query must be the side {@link levenshteinUniform} would itself have made
 * the pattern. Masks for the other side would have to be rebuilt for every
 * choice anyway, which is the cost being avoided.
 */
function preparedDistanceWorthwhile(
  uniform: boolean,
  queryLength: number,
  choiceLength: number,
  scoreCutoff: number,
  scoreHint: number,
): boolean {
  if (!uniform || queryLength === 0 || choiceLength === 0) return false

  const longest = Math.max(queryLength, choiceLength)
  if (Math.floor(scoreCutoff) < longest) return false

  // Mirrors the widening loop's starting width: below the cutoff it runs banded
  // passes first, and those are the ones the held pattern cannot serve.
  const difference = Math.abs(queryLength - choiceLength)
  if (Math.max(difference, Math.floor(scoreHint), 31) < longest) return false

  const queryWords = (queryLength + 31) >>> 5
  const choiceWords = (choiceLength + 31) >>> 5
  return queryWords * choiceLength <= choiceWords * queryLength
}

/**
 * Whether the held pattern can serve a budgeted score through the small band.
 *
 * `extract` tightens its cutoff as the heap fills, so nearly every choice it
 * scores arrives with a budget — which is what kept it on the unprepared path
 * while `cdist` moved onto the held masks.
 *
 * {@link levenshteinSmallBand} reads its masks through `shiftedPatternMatches`,
 * which windows a pattern of any width, so the whole-query masks serve it
 * unchanged. What it will not do is fit a band wider than a word: the budget
 * has to satisfy `2 * budget + 1 <= 32`, or the kernel drops the part that does
 * not fit and answers too large. {@link MAX_BAND_BUDGET} is that bound, and the
 * caller applies it — it still covers what `extract` asks for, since a heap of
 * the best few over short text settles on a budget well inside it.
 *
 * The remaining bounds are the kernel's other two preconditions, plus a length
 * test that is only an early out.
 *
 * Budgets under four are left alone: `levenshteinMbleven` answers those by
 * comparing elements directly, and builds no masks to save.
 */
function preparedBandWorthwhile(
  queryLength: number,
  choiceLength: number,
  budget: number,
): boolean {
  if (!Number.isFinite(budget) || budget < 4) return false

  // Out of reach on length alone: `distance_` returns `budget + 1` for this
  // without touching a mask, so there is nothing to save.
  if (Math.abs(queryLength - choiceLength) > budget) return false

  return (
    budget <= queryLength &&
    budget <= choiceLength &&
    choiceLength >= queryLength - budget
  )
}

/**
 * Fewest words a pattern has to span before trimming an affix beats holding
 * its masks.
 *
 * The held pattern's whole advantage is the mask build it does not repeat per
 * choice; trimming's is the elements the kernel never reaches, and each of
 * those is worth one step *per word*. So the two scale differently, and below
 * four words trimming loses even when the affix is nearly the whole input: at
 * two words a pair sharing 56 of 64 elements measured 0.83x, and at three words
 * an affix of 84 in 96 bought 1.13x against 0.92x on the near-copies beside it.
 * Four words is where the win clears what the probe costs — 1.5x there, 2.6x at
 * eight words and 9.2x at sixteen.
 */
const AFFIX_TRIM_WORDS = 4

/**
 * Whether the pair is better served by trimming its common affix than by the
 * held pattern the length gate has just approved.
 *
 * {@link preparedDistanceWorthwhile} answers from lengths, and lengths cannot
 * see an affix: a query and a choice of 512 elements that agree on 480 of them
 * look exactly like two unrelated ones. The held pattern reads all 512 either
 * way, where the unprepared kernel trims first and scores 32.
 *
 * The same relaxation the LCS metrics take, in the opposite direction — there
 * the probe lets a pair *onto* the held masks that a length gate refused, here
 * it takes one *off* them. That is why it asks {@link sharesWideAffix} for a
 * quarter rather than {@link sharesAffix}'s eighth, and why a pattern narrower
 * than {@link AFFIX_TRIM_WORDS} is not asked at all.
 */
function worthTrimming(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  const words = (Math.min(a.length, b.length) + 31) >>> 5
  return words >= AFFIX_TRIM_WORDS && sharesWideAffix(a, b)
}

function prepareLevenshtein(kind: PreparedLevenshteinKind): PrepareScorer {
  const prepare: PrepareScorer = (query, kwargs) => {
    const weights = parseWeights(Reflect.get(kwargs, 'weights'))
    const a = preparedScorerSequence(prepareScorerChoice(query))
    if (a === null) throw new TypeError('expected a sequence')

    const [insert, delete_, replace] = weights
    const uniform = insert === 1 && delete_ === 1 && replace === 1
    const integral = integralWeights(weights)
    let pattern: import('./_bitVector/index.js').PatternMask | null = null

    const preparedDistance = (
      b: ArrayLike<unknown>,
      cutoff: number,
      hint: number,
    ): number => {
      // A scorer with no cutoff runs with `cutoff` at `MAX_SAFE_INTEGER`, which
      // is what `cdist` does, and this comparison keeps it from paying for the
      // rest of the band test. It is also the only place the band's width bound
      // is applied, so `preparedBandWorthwhile` can take the budget as given.
      if (cutoff < MAX_BAND_BUDGET + 1 && uniform && a.length > 0 && b.length > 0) {
        const budget = Math.floor(cutoff)
        if (preparedBandWorthwhile(a.length, b.length, budget)) {
          pattern ??= preparePattern(a, 0, a.length)
          return levenshteinSmallBand(pattern, a.length, b, 0, b.length, budget)
        }
      }
      if (
        preparedDistanceWorthwhile(uniform, a.length, b.length, cutoff, hint) &&
        !worthTrimming(a, b)
      ) {
        pattern ??= preparePattern(a, 0, a.length)
        return levenshteinPrepared(pattern, b, 0, b.length)
      }
      // The unprepared kernel trims a common affix, which compares the two
      // sequences elementwise, so they have to agree on how a character is
      // spelled. The held pattern above reads either representation.
      return distance_(
        alignRepresentation(a, b),
        alignRepresentation(b, a),
        weights,
        cutoff,
        hint,
      )
    }

    const score: PreparedScore = (rawChoice, rawCutoff, rawHint) => {
      if (isNone(rawChoice)) {
        if (kind === 'normalizedDistance') return 1
        if (kind === 'normalizedSimilarity') return 0
      }
      let b = preparedScorerSequence(rawChoice)
      if (b === null) {
        if (!isSequence(rawChoice)) {
          throw new TypeError('expected a string or an array-like sequence')
        }
        b = scorerSequence(rawChoice)
      }
      // Lengths are all `maximum` reads, and aligning cannot change them, so it
      // is left to the unprepared path in `preparedDistance` that needs it.
      const max = maximum(a, b, weights)
      switch (kind) {
        case 'distance': {
          const cutoff = levenshteinRawCutoff(rawCutoff, integral)
          const bound = cutoff ?? Number.MAX_SAFE_INTEGER
          const hint = rawHint ?? bound
          const distance = preparedDistance(b, bound, hint)
          return cutoff === null || distance <= cutoff ? distance : cutoff + 1
        }
        case 'similarity': {
          const cutoff = levenshteinRawCutoff(rawCutoff, integral)
          const bound =
            cutoff === null ? Number.MAX_SAFE_INTEGER : rawBound(max - cutoff, integral)
          const hint = rawHint === null ? bound : rawBound(max - rawHint, integral)
          const similarity = max - preparedDistance(b, bound, hint)
          return cutoff === null || similarity >= cutoff ? similarity : 0
        }
        case 'normalizedDistance': {
          const cutoff =
            rawCutoff === null
              ? Number.MAX_SAFE_INTEGER
              : rawBound(rawCutoff * max, integral)
          const hint = rawHint === null ? cutoff : rawBound(rawHint * max, integral)
          return normDistCutoff(
            normalize(preparedDistance(b, cutoff, hint), max),
            rawCutoff,
          )
        }
        case 'normalizedSimilarity': {
          const cutoff =
            rawCutoff === null
              ? Number.MAX_SAFE_INTEGER
              : rawBound((1 - rawCutoff) * max, integral)
          const hint = rawHint === null ? cutoff : rawBound((1 - rawHint) * max, integral)
          return normSimCutoff(
            1 - normalize(preparedDistance(b, cutoff, hint), max),
            rawCutoff,
          )
        }
      }
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

/**
 * Weighted Levenshtein is symmetric only when insertion and deletion cost the
 * same, because swapping the arguments swaps those two operations.
 *
 * `configure` calls this when weights are baked into a scorer, which is the
 * only way they can reach a matrix now. Reporting it through the flags is what
 * lets `cdist` decide whether it may mirror the lower triangle without knowing
 * that the option in question is spelled `weights`.
 */
function levenshteinConfiguredFlags(base: ScorerFlags): ConfiguredFlags {
  return (options) => {
    const { insertion, deletion } = levenshteinCosts(Reflect.get(options, 'weights'))
    return insertion === deletion ? base : { ...base, symmetric: false }
  }
}

/**
 * Snapshot baked `weights` so the caller can no longer reach them.
 *
 * `configure` records the symmetry of a weighting once, at the moment it is
 * baked in. Both spellings of the option are mutable — an array and an object
 * of three numbers — so without this a caller could bake a symmetric weighting,
 * mutate it afterwards, and leave a scorer that scores asymmetrically while its
 * recorded flags still permit `scoreMatrix` to mirror half the matrix. That is
 * a wrong number, not a stale one.
 *
 * {@link levenshteinCosts} already allocates, so the copy is what it returns;
 * freezing it costs one call, once per `configure`.
 */
const levenshteinConfigureOptions: ConfigureOptions = (options) => {
  const weights = Reflect.get(options, 'weights')
  if (weights == null) return options
  return { ...options, weights: Object.freeze(levenshteinCosts(weights)) }
}

// Scorer flags let `process` tell distances from similarities.
export const levenshteinDistance: Scorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinDistance_impl,
    DISTANCE_FLAGS,
    prepareLevenshtein('distance'),
    {
      configuredFlags: /* @__PURE__ */ levenshteinConfiguredFlags(DISTANCE_FLAGS),
      configureOptions: levenshteinConfigureOptions,
    },
  )
export const levenshteinSimilarity: Scorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareLevenshtein('similarity'),
    {
      configuredFlags: /* @__PURE__ */ levenshteinConfiguredFlags(SIMILARITY_FLAGS),
      configureOptions: levenshteinConfigureOptions,
    },
  )
export const levenshteinNormalizedDistance: NormalizedScorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareLevenshtein('normalizedDistance'),
    {
      configuredFlags: /* @__PURE__ */ levenshteinConfiguredFlags(
        NORMALIZED_DISTANCE_FLAGS,
      ),
      configureOptions: levenshteinConfigureOptions,
    },
  )
export const levenshteinNormalizedSimilarity: NormalizedScorer<LevenshteinOptions> =
  /* @__PURE__ */ withPreparedFlags(
    levenshteinNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareLevenshtein('normalizedSimilarity'),
    {
      configuredFlags: /* @__PURE__ */ levenshteinConfiguredFlags(
        NORMALIZED_SIMILARITY_FLAGS,
      ),
      configureOptions: levenshteinConfigureOptions,
    },
  )
