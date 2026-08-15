import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { commonAffix } from '../shared/affix.js'
import { WORD_LIMIT } from '../shared/bitmask/blockMasks.js'
import { preparePattern } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  convPair,
  distanceCutoffFor,
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

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
  return Math.max(s1.length, s2.length)
}

function osaDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const cutoff = distanceCutoffFor('distance', options.scoreCutoff, maximum(a, b))
  return distCutoff(distance_(a, b, cutoff), options.scoreCutoff)
}

function osaSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = distanceCutoffFor('similarity', options.scoreCutoff, max)
  return simCutoff(max - distance_(a, b, cutoff), options.scoreCutoff)
}

function osaNormalizedDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = distanceCutoffFor('normalizedDistance', options.scoreCutoff, max)
  return normDistCutoff(
    normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

function osaNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = distanceCutoffFor('normalizedSimilarity', options.scoreCutoff, max)
  return normSimCutoff(
    1 - normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

type PreparedOsaKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareOsa(kind: PreparedOsaKind): PreparationFactory {
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
            a.length <= b.length
            ? a.length <= WORD_LIMIT
              ? osaOneWordPrepared(pattern, b)
              : osaPrepared(pattern, b)
            : distance_(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)

      switch (kind) {
        case 'distance':
          return distCutoff(distance, rawCutoff)
        case 'similarity':
          return simCutoff(max - distance, rawCutoff)
        case 'normalizedDistance':
          return normDistCutoff(normalizeDistance(distance, max), rawCutoff)
        case 'normalizedSimilarity':
          return normSimCutoff(1 - normalizeDistance(distance, max), rawCutoff)
      }
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

export const osaDistance: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  osaDistance_impl,
  DISTANCE_FLAGS,
  prepareOsa('distance'),
)
export const osaSimilarity: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  osaSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareOsa('similarity'),
)
export const osaNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    osaNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareOsa('normalizedDistance'),
  )
export const osaNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    osaNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareOsa('normalizedSimilarity'),
  )
