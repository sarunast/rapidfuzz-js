import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { commonAffix, sharesAffix } from '../shared/affix.js'
import { UNBOUNDED_MISSES, wordCount } from '../shared/bitmask/blockMasks.js'
import { preparePattern } from '../shared/bitmask/pattern.js'
import { lcsSeqMatrix, rowBitSet } from '../shared/bitParallel.js'
import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '../shared/editops/index.js'
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
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  withPreparedFlags,
  type MetricImplementation,
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
  const words = wordCount(queryLength)
  const fullBand = queryLength + choiceLength - 2 * required + 1
  const activeWords = Math.min(words, Math.floor(fullBand / 32) + 2)
  return queryLength <= choiceLength && words <= activeWords * 2
}

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

function lcsSeqSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const cutoff = canonicalSimilarityCutoff(options.scoreCutoff)
  const misses = cutoff === null ? Number.MAX_SAFE_INTEGER : maximum(a, b) - cutoff
  return simCutoff(boundedLength(a, b, misses), cutoff)
}

function lcsSeqNormalizedDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? Number.MAX_SAFE_INTEGER : options.scoreCutoff * max
  return normDistCutoff(
    normalizeDistance(max - boundedLength(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

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
    1 - normalizeDistance(max - boundedLength(a, b, cutoff), max),
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

/**
 * {@link lcsSeqEditops} as contiguous ranges rather than single operations.
 *
 * Opcodes cover the whole of both inputs, including the `equal` stretches
 * between edits, which is usually what a diff view or a highlighter wants —
 * `editops` lists only the changes. The two convert into each other with
 * `toEditops()` and `toOpcodes()`.
 */
export function lcsSeqOpcodes(s1: Sequence, s2: Sequence): Opcodes {
  return lcsSeqEditops(s1, s2).toOpcodes()
}

type PreparedLcsKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareLcs(kind: PreparedLcsKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    let pattern: import('../shared/bitmask/pattern.js').PatternMask | null = null
    const length = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (!preparedLengthWorthwhile(a.length, b.length, cutoff) && sharesAffix(a, b)) {
        return boundedLength(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= preparePattern(a, 0, a.length)
      const required = Math.max(0, Math.floor(maximum(a, b) - cutoff))
      return required > 0
        ? lcsLengthPreparedBounded(pattern, b, 0, b.length, required)
        : lcsLengthPrepared(pattern, b, 0, b.length)
    }

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)
      const max = maximum(a, b)
      switch (kind) {
        case 'distance': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          return distCutoff(max - length(b, cutoff ?? Number.MAX_SAFE_INTEGER), cutoff)
        }
        case 'similarity': {
          const cutoff = canonicalSimilarityCutoff(rawCutoff)
          const misses = cutoff === null ? Number.MAX_SAFE_INTEGER : max - cutoff
          return simCutoff(length(b, misses), cutoff)
        }
        case 'normalizedDistance': {
          const cutoff = rawCutoff === null ? Number.MAX_SAFE_INTEGER : rawCutoff * max
          return normDistCutoff(
            normalizeDistance(max - length(b, cutoff), max),
            rawCutoff,
          )
        }
        case 'normalizedSimilarity': {
          const cutoff =
            rawCutoff === null ? Number.MAX_SAFE_INTEGER : (1 - rawCutoff) * max
          return normSimCutoff(
            1 - normalizeDistance(max - length(b, cutoff), max),
            rawCutoff,
          )
        }
      }
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

export const lcsSeqDistance: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  lcsSeqDistance_impl,
  DISTANCE_FLAGS,
  prepareLcs('distance'),
)
export const lcsSeqSimilarity: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  lcsSeqSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareLcs('similarity'),
)
export const lcsSeqNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    lcsSeqNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareLcs('normalizedDistance'),
  )
export const lcsSeqNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    lcsSeqNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareLcs('normalizedSimilarity'),
  )
