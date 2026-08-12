import type { PreparedKernel } from '../../core/protocol.js'
import {
  buildProfile,
  dotProduct,
  dotProductKernel,
  parseGramSize,
  preparedProfile,
  profileOfElements,
  validGramSize,
  zeroGramSimilarity,
  type FrequencyKernel,
  type NGramProfile,
} from '../shared/ngram.js'
import {
  asSequence,
  convPair,
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  normDistCutoff,
  normSimCutoff,
  withPreparedFlags,
  type ConfigurationCanonicalizer,
  type MaybeSequence,
  type MaybeSequenceMetricImplementation,
  type PreparationFactory,
  type ScorerOptions,
  type Sequence,
} from '../shared/scorerSupport.js'

export interface CosineOptions extends ScorerOptions {
  readonly gramSize?: number | undefined
}

/**
 * Unlike Dice, this branch is needed on both sides. With `‖A‖ = 0` and
 * `‖B‖ > 0` the quotient is `0 / 0`, which is `NaN` rather than `0`; a sequence
 * shorter than `gramSize` has no grams and no norm at all.
 */
function profileSimilarity(
  a: NGramProfile,
  dot: FrequencyKernel,
  b: NGramProfile,
  scoreCutoff: number,
): number {
  if (a.gramCount === 0 || b.gramCount === 0) {
    const similarity = zeroGramSimilarity(a, b)
    return similarity >= scoreCutoff ? similarity : 0
  }
  // One square root of the product rather than two of the factors:
  // `Math.sqrt(3) * Math.sqrt(3)` is 3.0000000000000004, which would leave a
  // profile scored against itself just short of 1. The clamp covers the same
  // rounding in the other direction.
  const similarity = Math.min(dot(b) / Math.sqrt(a.squaredNorm * b.squaredNorm), 1)
  return similarity >= scoreCutoff ? similarity : 0
}

/**
 * Cosine similarity over n-gram frequency vectors, in `[0, 1]`.
 *
 * The dot product of the two frequency vectors over the product of their
 * lengths — not the intersection-count formula some libraries ship under this
 * name. No padding is added at the ends.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function cosineSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: CosineOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const gramSize = validGramSize(options.gramSize)
  const scoreCutoff = options.scoreCutoff
  // `convPair`, not `convSequence` per side — see the Dice implementation: two
  // BMP strings stay strings, and the pair form is what keeps both profiles
  // keyed in one element domain.
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const query = profileOfElements(a, gramSize)
  // One comparison, so no kernel to amortize: the generic walk is what a direct
  // call wants.
  return normSimCutoff(
    profileSimilarity(
      query,
      (choice) => dotProduct(query, choice),
      profileOfElements(b, gramSize),
      scoreCutoff ?? 0,
    ),
    scoreCutoff,
  )
}

/** Cosine distance in `[0, 1]`, i.e. `1 - similarity`. */
function cosineDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: CosineOptions = {},
): number {
  const gramSize = validGramSize(options.gramSize)
  const cutoff = options.scoreCutoff
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  const query = profileOfElements(a, gramSize)
  return normDistCutoff(
    1 -
      profileSimilarity(
        query,
        (choice) => dotProduct(query, choice),
        profileOfElements(b, gramSize),
        cutoff == null ? 0 : 1 - cutoff,
      ),
    cutoff,
  )
}

type PreparedCosineKind = 'distance' | 'similarity'

function prepareCosine(kind: PreparedCosineKind): PreparationFactory {
  return (options) => {
    // Once per scorer, so a matcher preparing many queries never reparses it.
    const gramSize = parseGramSize(options)

    const prepareChoice = (choice: Sequence): NGramProfile =>
      buildProfile(choice, gramSize)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = buildProfile(query, gramSize)
      // The query's trie is walked once, here, and never again while the search
      // runs — see `dotProductKernel`.
      const dot = dotProductKernel(a)

      return (rawChoice, rawCutoff) => {
        const b = preparedProfile(rawChoice)
        const similarityCutoff =
          kind === 'distance'
            ? rawCutoff === null
              ? 0
              : 1 - rawCutoff
            : (rawCutoff ?? 0)
        const similarity = profileSimilarity(a, dot, b, similarityCutoff)
        return kind === 'distance'
          ? normDistCutoff(1 - similarity, rawCutoff)
          : normSimCutoff(similarity, rawCutoff)
      }
    }

    return { prepareQuery, prepareChoice }
  }
}

/**
 * Settle `gramSize` once, when a scorer is compiled, and drop it when it is the
 * default — see the Dice canonicalizer, which does the same for the same
 * reason: it is what lets `{ gramSize: 2 }` and no configuration at all share
 * one prepared-choice key.
 */
const cosineConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) =>
  parseGramSize(options) === 2 ? {} : options

export const cosineSimilarity: MaybeSequenceMetricImplementation<CosineOptions> =
  /* @__PURE__ */ withPreparedFlags(
    cosineSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareCosine('similarity'),
    { configurationCanonicalizer: cosineConfigurationCanonicalizer },
  )
export const cosineDistance: MaybeSequenceMetricImplementation<CosineOptions> =
  /* @__PURE__ */ withPreparedFlags(
    cosineDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareCosine('distance'),
    { configurationCanonicalizer: cosineConfigurationCanonicalizer },
  )
