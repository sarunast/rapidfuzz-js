import {
  editopsFromValidated,
  type Editop,
  type Editops,
  type Opcodes,
} from '#core/editops/index.js'
import {
  canonicalRawCutoff,
  canonicalSimilarityCutoff,
  distanceCutoffFor,
  scoreFromDistance,
  type MetricScoreKind,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
} from '#core/scoring/builtIn/cutoff.js'
import {
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  withPreparedFlags,
  type MetricImplementation,
} from '#core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '#core/scoring/builtIn/options.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  type PreparationFactory,
} from '#core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import {
  alignRepresentation,
  convPair,
  convSequence,
  scorerSequence,
  maxSequenceLength,
} from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { commonAffix, passesAffixProbe } from '../affix.js'
import { UNBOUNDED_MISSES } from '../bitmask/blockMasks.js'
import { preparePattern, type PatternMask } from '../bitmask/pattern.js'
import { rowBitSet } from '../bitmask/rowBits.js'
import { wordCount } from '../bitmask/words.js'
import {
  lcsLengthPrepared,
  lcsLengthPreparedBounded,
  lcsLengthRange,
} from './internal/kernel.js'
import { lcsSeqMatrix } from './internal/matrix.js'

export {
  lcsLengthPrepared as lcsSeqLengthPrepared,
  lcsLengthPreparedBounded as lcsSeqLengthPreparedBounded,
  lcsLengthRange as lcsSeqLengthRange,
  preparePattern as prepareLcsPattern,
  UNBOUNDED_MISSES,
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
  const max = maxSequenceLength(a, b)
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
  const misses =
    cutoff === null ? Number.MAX_SAFE_INTEGER : maxSequenceLength(a, b) - cutoff
  return simCutoff(boundedLength(a, b, misses), cutoff)
}

function lcsSeqNormalizedDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maxSequenceLength(a, b)
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
  const max = maxSequenceLength(a, b)
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

function prepareLcs(kind: MetricScoreKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    let pattern: PatternMask | null = null
    let convertedQuery: ArrayLike<unknown> | null = null
    const alignedQueryFor = (b: ArrayLike<unknown>): ArrayLike<unknown> =>
      typeof a === 'string' && typeof b !== 'string'
        ? (convertedQuery ??= convSequence(a))
        : a
    const length = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (
        !preparedLengthWorthwhile(a.length, b.length, cutoff) &&
        passesAffixProbe(a, b)
      ) {
        return boundedLength(alignedQueryFor(b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= preparePattern(a, 0, a.length)
      const required = Math.max(0, Math.floor(maxSequenceLength(a, b) - cutoff))
      return required > 0
        ? lcsLengthPreparedBounded(pattern, b, 0, b.length, required)
        : lcsLengthPrepared(pattern, b, 0, b.length)
    }

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)
      const max = maxSequenceLength(a, b)
      const budget = distanceCutoffFor(kind, rawCutoff, max, Number.MAX_SAFE_INTEGER)
      return scoreFromDistance(kind, max - length(b, budget), max, rawCutoff)
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
