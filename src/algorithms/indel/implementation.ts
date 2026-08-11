import {
  lcsSeqEditops,
  lcsSeqLengthPrepared,
  lcsSeqLengthPreparedBounded,
  lcsSeqLengthRange,
  prepareLcsPattern,
  UNBOUNDED_MISSES,
} from '../lcs/implementation.js'
import { sharesAffix } from '../shared/affix.js'
import type { Editops, Opcodes } from '../shared/editops/index.js'
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
  type PrepareScorer,
  type PreparedScorerFactory,
  type PreparedScore,
  withPreparedFlags,
  type NormalizedScorer,
  type Scorer,
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
  const words = (queryLength + 31) >>> 5
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
    Math.ceil((query.length + choice.length - scoreCutoff) / 2),
  )
  const lcs =
    required > 0
      ? lcsSeqLengthPreparedBounded(pattern, choice, 0, choice.length, required)
      : lcsSeqLengthPrepared(pattern, choice, 0, choice.length)
  // The bounded kernel gave up, so all this owes its caller is a distance the
  // cutoff rejects. `floor(scoreCutoff) + 1` is not that distance: the four
  // kinds below hand in four different cutoffs, and only `distance` hands in a
  // whole number of edits. `normalizedSimilarity` at 0.8333… over `'  '` and
  // `'c😁b😁'` asks for `(1 - cutoff) * 6`, which floating point spells
  // 0.9999999999999998, so the sentinel came back as 1 — read as a real
  // distance of one edit, normalised to exactly the cutoff, and reported as a
  // score for a pair with nothing in common. One past the largest distance two
  // sequences of these lengths can have is the value no cutoff can read back.
  if (lcs < 0) return query.length + choice.length + 1
  return query.length + choice.length - 2 * lcs
}

/**
 * Insert/delete edit distance: how many elements must be inserted or deleted to
 * turn `s1` into `s2`. Substitutions are not allowed, which makes this
 * `|s1| + |s2| - 2 * LCS(s1, s2)`.
 *
 * If the distance is greater than `scoreCutoff`, `scoreCutoff + 1` is returned.
 */
function indelDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  return distCutoff(distance_(a, b, cutoff ?? UNBOUNDED_MISSES), cutoff)
}

/**
 * Indel similarity: `|s1| + |s2| - indelDistance(s1, s2)`.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function indelSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: ScorerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const max = maximum(a, b)
  const cutoff = canonicalRawCutoff(options.scoreCutoff)
  const misses = cutoff == null ? UNBOUNDED_MISSES : max - cutoff
  return simCutoff(max - distance_(a, b, misses), cutoff)
}

/**
 * {@link indelDistance} normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function indelNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? UNBOUNDED_MISSES : options.scoreCutoff * max
  return normDistCutoff(normalize(distance_(a, b, cutoff), max), options.scoreCutoff)
}

/**
 * {@link indelSimilarity} normalised into `[0, 1]`, where `1` means identical.
 * Two empty inputs are defined as identical.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function indelNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: ScorerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  const max = maximum(a, b)
  const cutoff =
    options.scoreCutoff == null ? UNBOUNDED_MISSES : (1 - options.scoreCutoff) * max
  return normSimCutoff(1 - normalize(distance_(a, b, cutoff), max), options.scoreCutoff)
}

/**
 * Edit operations that turn `s1` into `s2`. Identical to
 * {@link import('./lcsSeq.js').lcsSeqEditops} — the Indel metric is the LCS
 * metric counted differently.
 */
export function indelEditops(
  s1: Sequence,
  s2: Sequence,
  options: EditopsOptions = {},
): Editops {
  return lcsSeqEditops(s1, s2, options)
}

/** {@link indelEditops} expressed as blocks. */
export function indelOpcodes(
  s1: Sequence,
  s2: Sequence,
  options: EditopsOptions = {},
): Opcodes {
  return lcsSeqEditops(s1, s2, options).toOpcodes()
}

type PreparedIndelKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareIndel(kind: PreparedIndelKind): PreparedScorerFactory {
  const prepare: PrepareScorer = (query) => {
    const a = preparedScorerSequence(prepareScorerChoice(query))
    if (a === null) throw new TypeError('expected a sequence')
    let pattern: import('../shared/bitmask/pattern.js').PatternMask | null = null
    const preparedDistance = (b: ArrayLike<unknown>, cutoff: number): number => {
      if (!preparedDistanceWorthwhile(a.length, b.length, cutoff) && sharesAffix(a, b)) {
        // The unprepared kernel trims a common affix, which compares the two
        // sequences elementwise, so they have to agree on how a character is
        // spelled. The held pattern below reads either representation.
        return distance_(alignRepresentation(a, b), alignRepresentation(b, a), cutoff)
      }
      pattern ??= prepareLcsPattern(a, 0, a.length)
      return distanceFromPrepared(a, pattern, b, cutoff)
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
          return distCutoff(preparedDistance(b, cutoff ?? UNBOUNDED_MISSES), cutoff)
        }
        case 'similarity': {
          const cutoff = canonicalRawCutoff(rawCutoff)
          const misses = cutoff === null ? UNBOUNDED_MISSES : max - cutoff
          return simCutoff(max - preparedDistance(b, misses), cutoff)
        }
        case 'normalizedDistance': {
          const cutoff = rawCutoff === null ? UNBOUNDED_MISSES : rawCutoff * max
          return normDistCutoff(normalize(preparedDistance(b, cutoff), max), rawCutoff)
        }
        case 'normalizedSimilarity': {
          const cutoff = rawCutoff === null ? UNBOUNDED_MISSES : (1 - rawCutoff) * max
          return normSimCutoff(1 - normalize(preparedDistance(b, cutoff), max), rawCutoff)
        }
      }
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

// Scorer flags let `process` tell distances from similarities.
export const indelDistance: Scorer = /* @__PURE__ */ withPreparedFlags(
  indelDistance_impl,
  DISTANCE_FLAGS,
  prepareIndel('distance'),
)
export const indelSimilarity: Scorer = /* @__PURE__ */ withPreparedFlags(
  indelSimilarity_impl,
  SIMILARITY_FLAGS,
  prepareIndel('similarity'),
)
export const indelNormalizedDistance: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    indelNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareIndel('normalizedDistance'),
  )
export const indelNormalizedSimilarity: NormalizedScorer =
  /* @__PURE__ */ withPreparedFlags(
    indelNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareIndel('normalizedSimilarity'),
  )
