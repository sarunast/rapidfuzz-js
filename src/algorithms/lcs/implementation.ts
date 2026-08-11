import { sharesAffix } from '../shared/affix.js'
import { UNBOUNDED_MISSES } from '../shared/bitmask/blockMasks.js'
import { preparePattern } from '../shared/bitmask/pattern.js'
import { commonAffix, lcsSeqMatrix, rowBitSet } from '../shared/bitParallel.js'
import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '../shared/editops/index.js'
import {
  alignRepresentation,
  canonicalRawCutoff,
  convPair,
  distCutoff,
  normalize,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  withChoicePreparer,
  prepareScorerChoice,
  preparedScorerSequence,
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  withPreparedFlags,
  type Scorer,
} from '../shared/scorerSupport.js'
import {
  lcsLengthPrepared,
  lcsLengthPreparedBounded,
  lcsLengthRange,
} from './internal/kernel.js'

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
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  return distCutoff(max - boundedLength(a, b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
}

/**
 * LCS similarity normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function lcsSeqNormalizedSimilarity_impl(
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
export function lcsSeqEditops(s1: Sequence, s2: Sequence): Editops {
  const [full1, full2] = convPair(s1, s2)
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
export function lcsSeqOpcodes(s1: Sequence, s2: Sequence): Opcodes {
  return lcsSeqEditops(s1, s2).toOpcodes()
}

type PreparedLcsKind = 'distance' | 'normalizedSimilarity'

function prepareLcs(kind: PreparedLcsKind): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    let pattern: import('../shared/bitmask/pattern.js').PatternMask | null = null
    const length = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (!preparedLengthWorthwhile(a.length, b.length, cutoff) && sharesAffix(a, b)) {
        // The unprepared kernel trims a common affix, which compares the two
        // sequences elementwise, so they have to agree on how a character is
        // spelled. The held pattern below reads either representation.
        return boundedLength(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= preparePattern(a, 0, a.length)
      // A lower integer bound is deliberately conservative. Rounding a
      // normalized threshold back into edit units can land one ULP above an
      // integer; rounding upward there would reject a score exactly at the
      // caller's threshold. Being one match looser only costs pruning.
      const required = Math.max(0, Math.floor(maximum(a, b) - cutoff))
      return required > 0
        ? lcsLengthPreparedBounded(pattern, b, 0, b.length, required)
        : lcsLengthPrepared(pattern, b, 0, b.length)
    }

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const b = preparedScorerSequence(rawChoice)
      const max = maximum(a, b)
      switch (kind) {
        case 'distance': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          return distCutoff(max - length(b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
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

export const lcsSeqDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  lcsSeqDistance_impl,
  DISTANCE_FLAGS,
  prepareLcs('distance'),
)
export const lcsSeqNormalizedSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  lcsSeqNormalizedSimilarity_impl,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareLcs('normalizedSimilarity'),
)
