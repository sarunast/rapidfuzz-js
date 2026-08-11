import { WORD_LIMIT } from '../shared/bitmask/blockMasks.js'
import { preparePattern } from '../shared/bitmask/pattern.js'
import { commonAffix } from '../shared/bitParallel.js'
import {
  alignRepresentation,
  convPair,
  distanceCutoffFor,
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
import { osaOneWordRange, osaOneWordPrepared, osaPrepared } from './internal/kernel.js'

/**
 * Optimal String Alignment distance — Levenshtein plus transposition of two
 * *adjacent* elements, with the restriction that no substring is edited more
 * than once. That restriction is what separates OSA from Damerau-Levenshtein:
 * `osaDistance('CA', 'ABC')` is 3 where `damerauLevenshteinDistance` is 2.
 */
function distance_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  cutoff = Number.POSITIVE_INFINITY,
): number {
  const len1 = s1.length
  const len2 = s2.length

  if (len1 === 0) return len2
  if (len2 === 0) return len1

  // Every edit OSA can make changes the length by at most one, so the length
  // difference is a lower bound on the distance. During search this
  // rejects the badly sized candidates for the cost of a subtraction, without
  // the affix scan or the kernel running at all.
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

/**
 * Optimal String Alignment distance.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
function osaDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const cutoff = distanceCutoffFor('distance', options.scoreCutoff, maximum(a, b))
  return distCutoff(distance_(a, b, cutoff), options.scoreCutoff)
}

/**
 * OSA similarity normalised into `[0, 1]`, where `1` means identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function osaNormalizedSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = convPair(s1, s2)
  const max = maximum(a, b)
  const cutoff = distanceCutoffFor('normalizedSimilarity', options.scoreCutoff, max)
  return normSimCutoff(1 - normalize(distance_(a, b, cutoff), max), options.scoreCutoff)
}

type PreparedOsaKind = 'distance' | 'normalizedSimilarity'

function prepareOsa(kind: PreparedOsaKind): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const b = preparedScorerSequence(rawChoice)

      const max = Math.max(a.length, b.length)
      const cutoff = distanceCutoffFor(kind, rawCutoff, max)

      // The length difference is a lower bound on the distance, so a candidate
      // too differently sized to clear the cutoff is rejected before either
      // kernel is entered. `distCutoff` and friends below map the bail-out
      // value onto whatever this convention reports for a rejection.
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

      switch (kind) {
        case 'distance':
          return distCutoff(distance, rawCutoff)
        case 'normalizedSimilarity':
          return normSimCutoff(1 - normalize(distance, max), rawCutoff)
      }
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

export const osaDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  osaDistance_impl,
  DISTANCE_FLAGS,
  prepareOsa('distance'),
)
export const osaNormalizedSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  osaNormalizedSimilarity_impl,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareOsa('normalizedSimilarity'),
)
