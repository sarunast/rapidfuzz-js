import { jaroSimilarity_, jaroSimilarityPrepared_ } from '../jaro/implementation.js'
import { commonPrefix } from '../shared/affix.js'
import { preparePattern, type PatternMask } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  conv,
  normSimCutoff,
  type ScorerOptions,
  type Sequence,
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
  scoreCutoff: number,
  prepared: PatternMask,
): number {
  // No range test on `prefixWeight`: `prepareJaroWinkler` parses and checks it
  // once per query, before it builds the pattern this takes. `directSimilarity`
  // carries the test for the unprepared path, which has no such factory.
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

function prepareJaroWinkler(): PreparedScorerFactory {
  const prepare: PrepareScorer = (query, kwargs) => {
    const rawPrefixWeight = Reflect.get(kwargs, 'prefixWeight')
    const prefixWeight = rawPrefixWeight == null ? 0.1 : rawPrefixWeight
    if (typeof prefixWeight !== 'number')
      throw new TypeError('prefixWeight must be a number')
    if (prefixWeight > 1 || prefixWeight < 0) {
      throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
    }
    const a = preparedScorerSequence(prepareScorerChoice(query))
    const pattern = preparePattern(a, 0, a.length)

    const score: PreparedScore = (rawChoice, rawCutoff) => {
      const b = preparedScorerSequence(rawChoice)
      // The common-prefix bonus and Jaro's transposition pass both compare the
      // two sequences elementwise, so they have to agree on how a character is
      // spelled.
      const similarity = similarity_(
        alignRepresentation(a, b),
        alignRepresentation(b, a),
        prefixWeight,
        rawCutoff ?? 0,
        pattern,
      )
      return normSimCutoff(similarity, rawCutoff)
    }
    return score
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

export const jaroWinklerSimilarity: Scorer<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaroWinkler(),
  )
