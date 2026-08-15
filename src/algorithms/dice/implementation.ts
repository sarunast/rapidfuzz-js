import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { directSharedFrequency, sharedFrequency } from '../ngram/compare.js'
import { parseGramSize, validGramSize } from '../ngram/gramSize.js'
import { createDiceIndexBuilder } from '../ngram/inverted/dice.js'
import { sharedFrequencyKernel, type BoundedFrequencyKernel } from '../ngram/kernel.js'
import {
  buildProfile,
  preparedProfile,
  profileOfElements,
  zeroGramSimilarity,
  type NGramProfile,
} from '../ngram/profile.js'
import {
  validateSequence,
  convPair,
  elementsEqual,
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

export interface DiceOptions extends ScorerOptions {
  readonly gramSize?: number | undefined
}

function similarityBound(gramsA: number, gramsB: number): number {
  return (2 * Math.min(gramsA, gramsB)) / (gramsA + gramsB)
}

const COUNTER_GRAMS = 512
const COUNTER_GRAM_SIZES: readonly number[] = [2, 3]

function directSimilarity(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  gramSize: number,
  scoreCutoff: number,
): number {
  const gramsA = Math.max(0, a.length - gramSize + 1)
  const gramsB = Math.max(0, b.length - gramSize + 1)
  if (gramsA === 0 || gramsB === 0) {
    const similarity = gramsA === 0 && gramsB === 0 && elementsEqual(a, b) ? 1 : 0
    return similarity >= scoreCutoff ? similarity : 0
  }
  if (similarityBound(gramsA, gramsB) < scoreCutoff) return 0
  const counted =
    COUNTER_GRAM_SIZES.includes(gramSize) && Math.max(gramsA, gramsB) >= COUNTER_GRAMS
      ? directSharedFrequency(a, b, gramSize)
      : null
  const shared =
    counted ??
    sharedFrequency(profileOfElements(a, gramSize), profileOfElements(b, gramSize))
  const similarity = (2 * shared) / (gramsA + gramsB)
  return similarity >= scoreCutoff ? similarity : 0
}

function relaxedShared(denominator: number, scoreCutoff: number): number {
  return Math.ceil((scoreCutoff * denominator) / 2) - 2
}

function preparedSimilarity(
  a: NGramProfile,
  shared: BoundedFrequencyKernel,
  b: NGramProfile,
  scoreCutoff: number,
): number {
  const gramsA = a.gramCount
  const gramsB = b.gramCount
  if (gramsA === 0 || gramsB === 0) {
    const similarity = zeroGramSimilarity(a, b)
    return similarity >= scoreCutoff ? similarity : 0
  }
  if (similarityBound(gramsA, gramsB) < scoreCutoff) return 0
  const denominator = gramsA + gramsB
  const minimumShared = scoreCutoff > 0 ? relaxedShared(denominator, scoreCutoff) : 0
  const similarity = (2 * shared(b, minimumShared)) / denominator
  return similarity >= scoreCutoff ? similarity : 0
}

function diceSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: DiceOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const gramSize = validGramSize(options.gramSize)
  const scoreCutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normSimCutoff(directSimilarity(a, b, gramSize, scoreCutoff ?? 0), scoreCutoff)
}

function diceDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: DiceOptions = {},
): number {
  const gramSize = validGramSize(options.gramSize)
  const cutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normDistCutoff(
    1 - directSimilarity(a, b, gramSize, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

type PreparedDiceKind = 'distance' | 'similarity'

function prepareDice(kind: PreparedDiceKind): PreparationFactory {
  return (options) => {
    const gramSize = parseGramSize(options)

    const prepareChoice = (choice: Sequence): NGramProfile =>
      buildProfile(choice, gramSize)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = buildProfile(query, gramSize)
      const shared = sharedFrequencyKernel(a)

      return (rawChoice, rawCutoff) => {
        const b = preparedProfile(rawChoice)
        const similarityCutoff =
          kind === 'distance'
            ? rawCutoff === null
              ? 0
              : 1 - rawCutoff
            : (rawCutoff ?? 0)
        const similarity = preparedSimilarity(a, shared, b, similarityCutoff)
        return kind === 'distance'
          ? normDistCutoff(1 - similarity, rawCutoff)
          : normSimCutoff(similarity, rawCutoff)
      }
    }

    const indexChoices =
      kind === 'similarity' ? () => createDiceIndexBuilder(gramSize) : undefined

    return { prepareQuery, prepareChoice, indexChoices }
  }
}

const diceConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) =>
  parseGramSize(options) === 2 ? {} : options

export const diceSimilarity: MaybeSequenceMetricImplementation<DiceOptions> =
  /* @__PURE__ */ withPreparedFlags(
    diceSimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareDice('similarity'),
    { configurationCanonicalizer: diceConfigurationCanonicalizer },
  )
export const diceDistance: MaybeSequenceMetricImplementation<DiceOptions> =
  /* @__PURE__ */ withPreparedFlags(
    diceDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareDice('distance'),
    { configurationCanonicalizer: diceConfigurationCanonicalizer },
  )
