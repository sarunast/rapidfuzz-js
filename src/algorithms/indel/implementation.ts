import type { Editops, Opcodes } from '#core/editops/index.js'
import {
  distanceCutoffFor,
  scoreFromDistance,
  type MetricScoreKind,
} from '#core/scoring/builtIn/cutoff.js'
import { directMetric } from '#core/scoring/builtIn/directMetric.js'
import {
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  withPreparedFlags,
  type MetricImplementation,
} from '#core/scoring/builtIn/implementation.js'
import {
  prepareChoiceSequence,
  preparedChoiceSequence,
  type PreparationFactory,
} from '#core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import { alignRepresentation, queryAligner, scorerSequence } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { passesAffixProbe } from '../affix.js'
import type { PatternMask } from '../bitmask/pattern.js'
import { wordCount } from '../bitmask/words.js'
import {
  lcsSeqEditops,
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  lcsSeqLengthRange,
  prepareLcsPattern,
  UNBOUNDED_MISSES,
} from '../lcs/implementation.js'
import { createIndelCandidateIndexBuilder } from './candidateIndex.js'

function combinedLength(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return s1.length + s2.length
}

function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  scoreCutoff = UNBOUNDED_MISSES,
): number {
  const lcs = lcsSeqLengthRange(
    s1,
    0,
    s1.length,
    s2,
    0,
    s2.length,
    Math.floor(scoreCutoff),
  )
  return s1.length + s2.length - 2 * lcs
}

function preparedDistanceWorthwhile(
  queryLength: number,
  choiceLength: number,
  scoreCutoff: number,
): boolean {
  const required = Math.max(0, Math.ceil((queryLength + choiceLength - scoreCutoff) / 2))
  const words = wordCount(queryLength)
  const fullBand = queryLength + choiceLength - 2 * required + 1
  const activeWords = Math.min(words, Math.floor(fullBand / 32) + 2)
  return queryLength <= choiceLength && words <= activeWords * 2
}

function distanceFromPrepared(
  query: ArrayLike<unknown>,
  pattern: PatternMask,
  choice: ArrayLike<unknown>,
  scoreCutoff: number,
): number {
  const required = Math.max(
    0,
    Math.floor((query.length + choice.length - scoreCutoff) / 2),
  )
  const lcs =
    required > 0
      ? lcsSeqLengthPreparedBounded(pattern, choice, 0, choice.length, required)
      : lcsSeqLengthPrepared(pattern, choice, 0, choice.length)
  if (lcs < 0) return query.length + choice.length + 1
  return query.length + choice.length - 2 * lcs
}

function indelMetric(kind: MetricScoreKind) {
  return directMetric(kind, distance_, combinedLength, UNBOUNDED_MISSES)
}

/**
 * Edit operations that turn `s1` into `s2`. Identical to `lcsSeqEditops` — the
 * Indel metric is the LCS metric counted differently.
 */
export function indelEditops(s1: Sequence, s2: Sequence): Editops {
  return lcsSeqEditops(s1, s2)
}

/**
 * {@link indelEditops} as contiguous ranges rather than single operations.
 *
 * Opcodes cover the whole of both inputs, including the `equal` stretches
 * between edits, which is usually what a diff view or a highlighter wants —
 * `editops` lists only the changes. The two convert into each other with
 * `toEditops()` and `toOpcodes()`.
 */
export function indelOpcodes(s1: Sequence, s2: Sequence): Opcodes {
  return lcsSeqEditops(s1, s2).toOpcodes()
}

function prepareIndel(kind: MetricScoreKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    let pattern: PatternMask | null = null
    const alignedQueryFor = queryAligner(a)
    const preparedDistance = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (
        !preparedDistanceWorthwhile(a.length, b.length, cutoff) &&
        passesAffixProbe(a, b)
      ) {
        return distance_(alignedQueryFor(b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= prepareLcsPattern(a, 0, a.length)
      return distanceFromPrepared(a, pattern, b, cutoff)
    }

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)
      const max = combinedLength(a, b)
      const budget = distanceCutoffFor(kind, rawCutoff, max, UNBOUNDED_MISSES)
      return scoreFromDistance(kind, preparedDistance(b, budget), max, rawCutoff)
    }
    return score
  }
  return () => ({
    prepareQuery,
    prepareChoice: prepareChoiceSequence,
    candidateChoices:
      kind === 'normalizedSimilarity'
        ? () => createIndelCandidateIndexBuilder(prepareQuery)
        : undefined,
  })
}

export const indelDistance: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  indelMetric('distance'),
  DISTANCE_FLAGS,
  prepareIndel('distance'),
)
export const indelSimilarity: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  indelMetric('similarity'),
  SIMILARITY_FLAGS,
  prepareIndel('similarity'),
)
export const indelNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    indelMetric('normalizedDistance'),
    NORMALIZED_DISTANCE_FLAGS,
    prepareIndel('normalizedDistance'),
  )
export const indelNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    indelMetric('normalizedSimilarity'),
    NORMALIZED_SIMILARITY_FLAGS,
    prepareIndel('normalizedSimilarity'),
  )
