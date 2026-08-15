import type { PreparedKernel } from '../../core/scoring/compilation.js'
import {
  lcsSeqEditops,
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  lcsSeqLengthRange,
  prepareLcsPattern,
  UNBOUNDED_MISSES,
} from '../lcs/implementation.js'
import { sharesAffix } from '../shared/affix.js'
import { wordCount } from '../shared/bitmask/blockMasks.js'
import type { Editops, Opcodes } from '../shared/editops/index.js'
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

function maximum(s1: ArrayLike<unknown>, s2: ArrayLike<unknown>): number {
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
  pattern: import('../shared/bitmask/pattern.js').PatternMask,
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

function indelDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  return distCutoff(distance_(a, b, cutoff ?? UNBOUNDED_MISSES), cutoff)
}

function indelSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = canonicalSimilarityCutoff(options.scoreCutoff)
  const misses = cutoff == null ? UNBOUNDED_MISSES : max - cutoff
  return simCutoff(max - distance_(a, b, misses), cutoff)
}

function indelNormalizedDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? UNBOUNDED_MISSES : options.scoreCutoff * max
  return normDistCutoff(
    normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
}

function indelNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? UNBOUNDED_MISSES : (1 - options.scoreCutoff) * max
  return normSimCutoff(
    1 - normalizeDistance(distance_(a, b, cutoff), max),
    options.scoreCutoff,
  )
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

type PreparedIndelKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareIndel(kind: PreparedIndelKind): PreparationFactory {
  const prepareQuery = (query: Sequence): PreparedKernel => {
    const a = scorerSequence(query)
    let pattern: import('../shared/bitmask/pattern.js').PatternMask | null = null
    const preparedDistance = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (!preparedDistanceWorthwhile(a.length, b.length, cutoff) && sharesAffix(a, b)) {
        return distance_(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= prepareLcsPattern(a, 0, a.length)
      return distanceFromPrepared(a, pattern, b, cutoff)
    }

    const score: PreparedKernel = (rawChoice, rawCutoff) => {
      const b = preparedChoiceSequence(rawChoice)
      const max = maximum(a, b)
      switch (kind) {
        case 'distance': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          return distCutoff(preparedDistance(b, cutoff ?? UNBOUNDED_MISSES), cutoff)
        }
        case 'similarity': {
          const cutoff = canonicalSimilarityCutoff(rawCutoff)
          const misses = cutoff === null ? UNBOUNDED_MISSES : max - cutoff
          return simCutoff(max - preparedDistance(b, misses), cutoff)
        }
        case 'normalizedDistance': {
          const cutoff = rawCutoff === null ? UNBOUNDED_MISSES : rawCutoff * max
          return normDistCutoff(
            normalizeDistance(preparedDistance(b, cutoff), max),
            rawCutoff,
          )
        }
        case 'normalizedSimilarity': {
          const cutoff = rawCutoff === null ? UNBOUNDED_MISSES : (1 - rawCutoff) * max
          return normSimCutoff(
            1 - normalizeDistance(preparedDistance(b, cutoff), max),
            rawCutoff,
          )
        }
      }
    }
    return score
  }
  return () => ({ prepareQuery, prepareChoice: prepareChoiceSequence })
}

export const indelDistance: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  indelDistance_impl,
  DISTANCE_FLAGS,
  prepareIndel('distance'),
)
export const indelSimilarity: MetricImplementation = /* @__PURE__ */ withPreparedFlags(
  indelSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareIndel('similarity'),
)
export const indelNormalizedDistance: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    indelNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareIndel('normalizedDistance'),
  )
export const indelNormalizedSimilarity: MetricImplementation =
  /* @__PURE__ */ withPreparedFlags(
    indelNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareIndel('normalizedSimilarity'),
  )
