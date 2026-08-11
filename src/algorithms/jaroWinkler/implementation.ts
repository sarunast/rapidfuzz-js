import type { PreparedKernel } from '../../core/protocol.js'
import { jaroSimilarity_, jaroSimilarityPrepared_ } from '../jaro/implementation.js'
import { commonPrefix } from '../shared/affix.js'
import { preparePattern, type PatternMask } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  asSequence,
  convPair,
  type ConfigurationCanonicalizer,
  normDistCutoff,
  normSimCutoff,
  type MaybeSequence,
  type Sequence,
  type MaybeSequenceMetricImplementation,
  type ScorerOptions,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
  withPreparedFlags,
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
  // once per scorer, before it builds the pattern this takes. `directSimilarity`
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
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return normSimCutoff(
    directSimilarity(a, b, options.prefixWeight ?? 0.1, options.scoreCutoff ?? 0),
    options.scoreCutoff,
  )
}

/** Jaro-Winkler distance in `[0, 1]`, i.e. `1 - similarity`. */
function jaroWinklerDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: JaroWinklerOptions = {},
): number {
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
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

type PreparedJaroWinklerKind = 'distance' | 'similarity'

/**
 * One parse for the prepared factory and the canonicalizer, so they cannot
 * disagree. Testing for the range rather than against it rejects `NaN` with
 * the same comparison.
 */
function parsePrefixWeight(options: Readonly<Record<string, unknown>>): number {
  const prefixWeight = Reflect.get(options, 'prefixWeight')
  if (prefixWeight == null) return 0.1
  if (typeof prefixWeight !== 'number') {
    throw new TypeError('prefixWeight must be a number')
  }
  if (!(prefixWeight >= 0 && prefixWeight <= 1)) {
    throw new RangeError('prefix_weight has to be in the range 0.0 - 1.0')
  }
  return prefixWeight
}

function prepareJaroWinkler(kind: PreparedJaroWinklerKind): PreparationFactory {
  return (options) => {
    // Once per scorer, so a matcher preparing many queries never reparses it.
    const prefixWeight = parsePrefixWeight(options)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = scorerSequence(query)
      const pattern = preparePattern(a, 0, a.length)

      return (rawChoice, rawCutoff) => {
        const b = preparedChoiceSequence(rawChoice)
        // The common-prefix bonus and Jaro's transposition pass both compare the
        // two sequences elementwise, so they have to agree on how a character is
        // spelled.
        const similarityCutoff =
          kind === 'distance'
            ? rawCutoff === null
              ? 0
              : 1 - rawCutoff
            : (rawCutoff ?? 0)
        const similarity = similarity_(
          alignRepresentation(a, b),
          alignRepresentation(b, a),
          prefixWeight,
          similarityCutoff,
          pattern,
        )
        return kind === 'distance'
          ? normDistCutoff(1 - similarity, rawCutoff)
          : normSimCutoff(similarity, rawCutoff)
      }
    }

    return { prepareQuery, prepareChoice: prepareChoiceSequence }
  }
}

/**
 * Settle `prefixWeight` once, when a scorer is compiled.
 *
 * Every later path reads the weight from the record this returns, so a
 * non-number that {@link directSimilarity}'s numeric comparisons would quietly
 * coerce — or a `NaN` that would poison every score — is rejected here, before
 * either the direct or the prepared path can see it.
 */
const jaroWinklerConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  parsePrefixWeight(options)
  return options
}

export const jaroWinklerSimilarity: MaybeSequenceMetricImplementation<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareJaroWinkler('similarity'),
    { configurationCanonicalizer: jaroWinklerConfigurationCanonicalizer },
  )
export const jaroWinklerDistance: MaybeSequenceMetricImplementation<JaroWinklerOptions> =
  /* @__PURE__ */ withPreparedFlags(
    jaroWinklerDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareJaroWinkler('distance'),
    { configurationCanonicalizer: jaroWinklerConfigurationCanonicalizer },
  )
