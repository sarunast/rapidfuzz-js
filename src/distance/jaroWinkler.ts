import {
  alignRepresentation,
  commonPrefix,
  conv,
  normDistCutoff,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
  DISTANCE_FLAGS,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  SIMILARITY_FLAGS,
  type MaybeSequence,
  isNone,
  asSequence,
  isSequence,
  PREPARE_CHOICE,
  prepareScorerChoice,
  preparedScorerSequence,
  scorerSequence,
  type PrepareScorer,
  type PreparedScore,
  withPreparedFlags,
  type NormalizedScorer,
  type Scorer,
} from '../_common.js'
import { preparePattern } from './_bitVector/index.js'
import { jaroSimilarity_, jaroSimilarityPrepared_ } from './jaro.js'

export interface JaroWinklerOptions extends ScorerOptions {
  /** Weight of the common prefix bonus. Must be in `[0, 1]`. Defaults to `0.1`. */
  prefixWeight?: number | undefined
}

/**
 * Jaro similarity plus a bonus for a common prefix of up to four elements.
 *
 * The bonus only applies above a Jaro similarity of 0.7, matching RapidFuzz.
 */
function similarity_(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  prefixWeight: number,
  scoreCutoff = 0,
  prepared = preparePattern(s1, 0, s1.length),
): number {
  if (prefixWeight > 1 || prefixWeight < 0) {
    throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
  }

  const prefix = Math.min(commonPrefix(s1, s2), 4)
  let jaroCutoff = scoreCutoff
  if (jaroCutoff > 0.7) {
    const prefixSimilarity = prefix * prefixWeight
    jaroCutoff =
      prefixSimilarity >= 1
        ? 0.7
        : Math.max(0.7, (prefixSimilarity - jaroCutoff) / (prefixSimilarity - 1))
  }
  let sim = jaroSimilarityPrepared_(s1, prepared, s2, jaroCutoff)

  if (sim > 0.7) {
    sim += prefix * prefixWeight * (1 - sim)
    sim = Math.min(sim, 1)
  }

  return sim >= scoreCutoff ? sim : 0
}

function directSimilarity(
  s1: ArrayLike<unknown>,
  s2: ArrayLike<unknown>,
  prefixWeight: number,
  scoreCutoff: number,
): number {
  if (prefixWeight > 1 || prefixWeight < 0) {
    throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
  }
  const prefix = Math.min(commonPrefix(s1, s2), 4)
  let jaroCutoff = scoreCutoff
  if (jaroCutoff > 0.7) {
    const prefixSimilarity = prefix * prefixWeight
    jaroCutoff =
      prefixSimilarity >= 1
        ? 0.7
        : Math.max(0.7, (prefixSimilarity - jaroCutoff) / (prefixSimilarity - 1))
  }
  let sim = jaroSimilarity_(s1, s2, jaroCutoff)
  if (sim > 0.7) sim = Math.min(sim + prefix * prefixWeight * (1 - sim), 1)
  return sim >= scoreCutoff ? sim : 0
}

/**
 * Jaro-Winkler similarity in `[0, 1]`, where `1` means identical.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 *
 * The score is already normalised, so `scoreCutoff` is read as a `double` in
 * `[0, 1]` rather than as a raw, element-counting cutoff — see
 * {@link jaroSimilarity}, which upstream treats the same way.
 *
 * @throws if `prefixWeight` is outside `[0, 1]`.
 */
function jaroWinklerSimilarity_impl(
  s1: Sequence,
  s2: Sequence,
  options: JaroWinklerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  return normSimCutoff(
    directSimilarity(a, b, options.prefixWeight ?? 0.1, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

/**
 * Jaro-Winkler distance, i.e. `1 - jaroWinklerSimilarity(s1, s2)`.
 *
 * If the distance is greater than `scoreCutoff`, `1` is returned. See
 * {@link jaroWinklerSimilarity} for why the clamp is the normalised one.
 */
function jaroWinklerDistance_impl(
  s1: Sequence,
  s2: Sequence,
  options: JaroWinklerOptions = {},
): number {
  const [a, b] = conv(s1, s2, options.processor)
  const cutoff = options.scoreCutoff
  return normDistCutoff(
    1 -
      directSimilarity(
        a,
        b,
        options.prefixWeight ?? 0.1,
        cutoff == null ? 0 : 1 - cutoff,
      ),
    cutoff,
  )
}

/**
 * Jaro-Winkler distance. Identical to {@link jaroWinklerDistance} — the metric
 * is already normalised into `[0, 1]`.
 *
 * If the normalised distance is greater than `scoreCutoff`, `1` is returned.
 */
function jaroWinklerNormalizedDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 1

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  return normDistCutoff(
    1 -
      directSimilarity(
        a,
        b,
        options.prefixWeight ?? 0.1,
        options.scoreCutoff == null ? 0 : 1 - options.scoreCutoff,
      ),
    options.scoreCutoff,
  )
}

/**
 * Jaro-Winkler similarity. Identical to {@link jaroWinklerSimilarity} — the
 * metric is already normalised into `[0, 1]`.
 *
 * If the normalised similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function jaroWinklerNormalizedSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  if (isNone(s1) || isNone(s2)) return 0

  const [a, b] = conv(asSequence(s1), asSequence(s2), options.processor)
  return normSimCutoff(
    directSimilarity(a, b, options.prefixWeight ?? 0.1, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

type PreparedJaroWinklerKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function prepareJaroWinkler(kind: PreparedJaroWinklerKind): PrepareScorer {
  return (query, kwargs) => {
    const rawPrefixWeight = Reflect.get(kwargs, 'prefixWeight')
    const prefixWeight = rawPrefixWeight == null ? 0.1 : rawPrefixWeight
    if (typeof prefixWeight !== 'number')
      throw new TypeError('prefixWeight must be a number')
    if (prefixWeight > 1 || prefixWeight < 0) {
      throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
    }
    const a = preparedScorerSequence(prepareScorerChoice(query))
    if (a === null) throw new TypeError('expected a sequence')
    const pattern = preparePattern(a, 0, a.length)

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
      const similarityCutoff =
        kind === 'distance' || kind === 'normalizedDistance'
          ? rawCutoff === null
            ? 0
            : 1 - rawCutoff
          : (rawCutoff ?? 0)
      // The common-prefix bonus and Jaro's transposition pass both compare the
      // two sequences elementwise, so they have to agree on how a character is
      // spelled.
      const similarity = similarity_(
        alignRepresentation(a, b),
        alignRepresentation(b, a),
        prefixWeight,
        similarityCutoff,
        pattern,
      )

      // Jaro's score is normalised, so `distance` and `normalizedDistance` are
      // the same metric read the same way — and likewise the two similarities.
      switch (kind) {
        case 'distance':
        case 'normalizedDistance':
          return normDistCutoff(1 - similarity, rawCutoff)
        case 'similarity':
        case 'normalizedSimilarity':
          return normSimCutoff(similarity, rawCutoff)
      }
    }
    Object.defineProperty(score, PREPARE_CHOICE, { value: prepareScorerChoice })
    return score
  }
}

// Scorer flags let `process` tell distances from similarities.
export const jaroWinklerSimilarity: Scorer<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerSimilarity_impl,
    SIMILARITY_FLAGS,
    prepareJaroWinkler('similarity'),
  )
export const jaroWinklerDistance: Scorer<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerDistance_impl,
    DISTANCE_FLAGS,
    prepareJaroWinkler('distance'),
  )
export const jaroWinklerNormalizedDistance: NormalizedScorer<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerNormalizedDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareJaroWinkler('normalizedDistance'),
  )
export const jaroWinklerNormalizedSimilarity: NormalizedScorer<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerNormalizedSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaroWinkler('normalizedSimilarity'),
  )
