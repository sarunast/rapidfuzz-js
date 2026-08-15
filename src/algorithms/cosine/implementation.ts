import { normDistCutoff, normSimCutoff } from '../../core/scoring/builtIn/cutoff.js'
import {
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  withPreparedFlags,
  type ConfigurationCanonicalizer,
  type MaybeSequenceMetricImplementation,
} from '../../core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '../../core/scoring/builtIn/options.js'
import type { PreparationFactory } from '../../core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { validateSequence, convPair } from '../../core/sequence.js'
import type { MaybeSequence, Sequence } from '../../core/types.js'
import { dotProduct } from '../ngram/compare.js'
import { parseGramSize, validGramSize } from '../ngram/gramSize.js'
import { createCosineIndexBuilder } from '../ngram/inverted/cosine.js'
import { dotProductKernel, type FrequencyKernel } from '../ngram/kernel.js'
import {
  buildProfile,
  preparedProfile,
  profileOfElements,
  zeroGramSimilarity,
  type NGramProfile,
} from '../ngram/profile.js'

export interface CosineOptions extends ScorerOptions {
  readonly gramSize?: number | undefined
}

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
  const similarity = Math.min(dot(b) / Math.sqrt(a.squaredNorm * b.squaredNorm), 1)
  return similarity >= scoreCutoff ? similarity : 0
}

function cosineSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: CosineOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const gramSize = validGramSize(options.gramSize)
  const scoreCutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  const query = profileOfElements(a, gramSize)
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

function cosineDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: CosineOptions = {},
): number {
  const gramSize = validGramSize(options.gramSize)
  const cutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
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
    const gramSize = parseGramSize(options)

    const prepareChoice = (choice: Sequence): NGramProfile =>
      buildProfile(choice, gramSize)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = buildProfile(query, gramSize)
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

    const indexChoices =
      kind === 'similarity' ? () => createCosineIndexBuilder(gramSize) : undefined

    return { prepareQuery, prepareChoice, indexChoices }
  }
}

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
