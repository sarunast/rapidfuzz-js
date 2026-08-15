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
import { alignRepresentation, scorerSequence, maxSequenceLength } from '#core/sequence.js'
import type { Sequence } from '#core/types.js'

import { commonAffix } from '../affix.js'
import { WORD_LIMIT } from '../bitmask/blockMasks.js'
import { preparePattern } from '../bitmask/pattern.js'
import { osaOneWordRange, osaOneWordPrepared, osaPrepared } from './internal/kernel.js'

function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  cutoff = Number.POSITIVE_INFINITY,
): number {
  const len1 = s1.length
  const len2 = s2.length

  if (len1 === 0) return len2
  if (len2 === 0) return len1

  if (Math.abs(len1 - len2) > cutoff) return cutoff + 1

  const { prefixLen, suffixLen } = commonAffix(s1, s2)
  const trimmed1 = len1 - prefixLen - suffixLen
  const trimmed2 = len2 - prefixLen - suffixLen
  if (trimmed1 === 0) return trimmed2
  if (trimmed2 === 0) return trimmed1

  return trimmed1 <= trimmed2
    ? osaRange(s1, prefixLen, trimmed1, s2, prefixLen, trimmed2)
    : osaRange(s2, prefixLen, trimmed2, s1, prefixLen, trimmed1)
}

function osaRange(
  pattern: ArrayLike<unknown>,
  patternStart: number,
  patternLength: number,
  text: ArrayLike<unknown>,
  textStart: number,
  textLength: number,
): number {
  if (patternLength <= WORD_LIMIT) {
    return osaOneWordRange(
      pattern,
      patternStart,
      patternLength,
      text,
      textStart,
      textLength,
    )
  }
  return osaPrepared(
    preparePattern(pattern, patternStart, patternLength),
    text,
    textStart,
    textLength,
  )
}

function osaMetric(kind: MetricScoreKind) {
  return directMetric(kind, distance_, maxSequenceLength, Number.POSITIVE_INFINITY)
}

function prepareOsa(kind: MetricScoreKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)

      const max = Math.max(a.length, b.length)
      const cutoff = distanceCutoffFor(kind, rawCutoff, max)

      const distance =
        Math.abs(a.length - b.length) > cutoff
          ? cutoff + 1
          : // The fallback below trims a common affix, which compares the two
            // sequences elementwise, so they have to agree on how a character
            // is spelled. The held-pattern kernels read either representation.
            a.length <= b.length
            ? a.length <= WORD_LIMIT
              ? osaOneWordPrepared(pattern, b)
              : osaPrepared(pattern, b)
            : distance_(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)

      return scoreFromDistance(kind, distance, max, rawCutoff)
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

export const osaDistance: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  osaMetric('distance'),
  DISTANCE_FLAGS,
  prepareOsa('distance'),
)
export const osaSimilarity: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  osaMetric('similarity'),
  SIMILARITY_FLAGS,
  prepareOsa('similarity'),
)
export const osaNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    osaMetric('normalizedDistance'),
    NORMALIZED_DISTANCE_FLAGS,
    prepareOsa('normalizedDistance'),
  )
export const osaNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    osaMetric('normalizedSimilarity'),
    NORMALIZED_SIMILARITY_FLAGS,
    prepareOsa('normalizedSimilarity'),
  )
