import {
  alignRepresentation,
  canonicalRawCutoff,
  conv,
  distCutoff,
  normalize,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
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
  sharesAffix,
  type PrepareScorer,
  type PreparedScore,
  withPreparedFlags,
  type NormalizedScorer,
  type Scorer,
} from '../_common.js'
import { commonAffix, lcsSeqMatrix, rowBitSet } from './_bitParallel.js'
import {
  lcsLengthPrepared,
  lcsLengthPreparedBounded,
  lcsLengthRange,
  preparePattern,
  UNBOUNDED_MISSES,
} from './_bitVector/index.js'
import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from './editops.js'

export {
  lcsLengthPrepared as lcsSeqLengthPrepared,
  lcsLengthPreparedBounded as lcsSeqLengthPreparedBounded,
  lcsLengthRange as lcsSeqLengthRange,
  preparePattern as prepareLcsPattern,
  UNBOUNDED_MISSES,
}

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function boundedLength(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  distanceCutoff: number,
): number {
  const lengthDifference = Math.abs(s1.length - s2.length)
  const missBudget = Math.max(0, Math.floor(2 * distanceCutoff - lengthDifference))
  return lcsLengthRange(s1, 0, s1.length, s2, 0, s2.length, missBudget)
}

function preparedLengthWorthwhile(
  queryLength: number,
  choiceLength: number,
  distanceCutoff: number,
): boolean {
  const required = Math.max(
    0,
    Math.ceil(Math.max(queryLength, choiceLength) - distanceCutoff),
  )
  const words = (queryLength + 31) >>> 5
  const fullBand = queryLength + choiceLength - 2 * required + 1
  const activeWords = Math.min(words, Math.floor(fullBand / 32) + 2)
  return queryLength <= choiceLength && words <= activeWords * 2
}

/**
 * Number of elements that are not part of the longest common subsequence,
 * i.e. `max(|s1|, |s2|) - lcsSeqSimilarity(s1, s2)`.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
function lcsSeqDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const max = maximum(a, b)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  return distCutoff(max - boundedLength(a, b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
}

/**
 * Length of the longest common subsequence of `s1` and `s2`.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function lcsSeqSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  const misses = cutoff == null ? Number.MAX_SAFE_INTEGER : maximum(a, b) - cutoff
  return simCutoff(boundedLength(a, b, misses), cutoff)
}

/**
 * {@link lcsSeqDistance} normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function lcsSeqNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? Number.MAX_SAFE_INTEGER : options.scoreCutoff * max
  return normDistCutoff(
    normalize(max - boundedLength(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

/**
 * {@link lcsSeqSimilarity} normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function lcsSeqNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null
      ? Number.MAX_SAFE_INTEGER
      : (1 - options.scoreCutoff) * max
  return normSimCutoff(
    1 - normalize(max - boundedLength(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

/**
 * Edit operations that turn `s1` into `s2` using only insertions and deletions.
 *
 * The alignment follows Hyyrö's bit-parallel algorithm, so the operations match
 * the ones upstream produces — several alignments can be optimal and this picks
 * the same one.
 */
export function lcsSeqEditops(
  s1: Sequence,
  s2: Sequence,
  options: EditopsOptions = {},
): Editops {
  const [full1, full2] = conv(s1, s2, options.processor)
  const { prefixLen, suffixLen } = commonAffix(full1, full2)

  // The trimmed middle is addressed through `prefixLen` rather than copied out
  // of each input: recovery only ever reads it, and the copies were two arrays
  // the size of the inputs per call.
  const aLength = full1.length - suffixLen - prefixLen
  const bLength = full2.length - suffixLen - prefixLen
  const { sim, rows, words } = lcsSeqMatrix(
    full1,
    prefixLen,
    aLength,
    full2,
    prefixLen,
    bLength,
  )

  const srcLen = aLength + prefixLen + suffixLen
  const destLen = bLength + prefixLen + suffixLen

  let dist = aLength + bLength - 2 * sim
  if (dist === 0) return editopsFromValidated([], srcLen, destLen)

  const ops = new Array<Editop>(dist)
  let col = aLength
  let row = bLength

  while (row !== 0 && col !== 0) {
    if (rowBitSet(rows, words, row - 1, col - 1)) {
      dist--
      col--
      ops[dist] = { tag: 'delete', srcPos: col + prefixLen, destPos: row + prefixLen }
    } else {
      row--

      if (row && !rowBitSet(rows, words, row - 1, col - 1)) {
        dist--
        ops[dist] = { tag: 'insert', srcPos: col + prefixLen, destPos: row + prefixLen }
      } else {
        col--
      }
    }
  }

  while (col !== 0) {
    dist--
    col--
    ops[dist] = { tag: 'delete', srcPos: col + prefixLen, destPos: row + prefixLen }
  }

  while (row !== 0) {
    dist--
    row--
    ops[dist] = { tag: 'insert', srcPos: col + prefixLen, destPos: row + prefixLen }
  }

  return editopsFromValidated(ops, srcLen, destLen)
}

/** {@link lcsSeqEditops} expressed as blocks. */
export function lcsSeqOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: EditopsOptions = {},
): Opcodes {
  return lcsSeqEditops(s1, s2, options).toOpcodes()
}

type PreparedLcsKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareLcs(kind: PreparedLcsKind): PrepareScorer {
  const prepare: PrepareScorer = (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    if (a === null) throw new TypeError('expected a sequence')
    let pattern: import('./_bitVector/index.js').PatternMask | null = null
    const length = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (!preparedLengthWorthwhile(a.length, b.length, cutoff) && sharesAffix(a, b)) {
        // The unprepared kernel trims a common affix, which compares the two
        // sequences elementwise, so they have to agree on how a character is
        // spelled. The held pattern below reads either representation.
        return boundedLength(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= preparePattern(a, 0, a.length)
      const required = Math.max(0, Math.ceil(maximum(a, b) - cutoff))
      return required > 0
        ? lcsLengthPreparedBounded(pattern, b, 0, b.length, required)
        : lcsLengthPrepared(pattern, b, 0, b.length)
    }

    const score: PreparedScore = (rawChoice, rawCutoff) => {
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
      const max = maximum(a, b)
      switch (kind) {
        case 'distance': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          return distCutoff(max - length(b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
        }
        case 'similarity': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          const misses = cutoff === null ? Number.MAX_SAFE_INTEGER : max - cutoff
          return simCutoff(length(b, misses), cutoff)
        }
        case 'normalizedDistance': {
          const cutoff = rawCutoff === null ? Number.MAX_SAFE_INTEGER : rawCutoff * max
          return normDistCutoff(normalize(max - length(b, cutoff), max), rawCutoff)
        }
        case 'normalizedSimilarity': {
          const cutoff =
            rawCutoff === null ? Number.MAX_SAFE_INTEGER : (1 - rawCutoff) * max
          return normSimCutoff(1 - normalize(max - length(b, cutoff), max), rawCutoff)
        }
      }
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

// Scorer flags let `process` tell distances from similarities.
export const lcsSeqDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  lcsSeqDistance_impl,
  DISTANCE_FLAGS,
  prepareLcs('distance'),
)
export const lcsSeqSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  lcsSeqSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareLcs('similarity'),
)
export const lcsSeqNormalizedDistance: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    lcsSeqNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareLcs('normalizedDistance'),
  )
export const lcsSeqNormalizedSimilarity: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    lcsSeqNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareLcs('normalizedSimilarity'),
  )
