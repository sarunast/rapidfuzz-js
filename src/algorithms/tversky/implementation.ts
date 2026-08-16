import { normDistCutoff, normSimCutoff } from '#core/scoring/builtIn/cutoff.js'
import {
  NORMALIZED_DISTANCE_FLAGS,
  NORMALIZED_SIMILARITY_FLAGS,
  withPreparedFlags,
  type ConfigurationCanonicalizer,
  type ConfigurationSymmetryResolver,
  type MaybeSequenceMetricImplementation,
} from '#core/scoring/builtIn/implementation.js'
import type { ScorerOptions } from '#core/scoring/builtIn/options.js'
import type { PreparationFactory } from '#core/scoring/builtIn/preparation.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import { validateSequence, convPair, elementsEqual } from '#core/sequence.js'
import type { MaybeSequence, Sequence } from '#core/types.js'

import { directSharedFrequency, sharedFrequency } from '../ngram/compare.js'
import { parseGramSize, validGramSize } from '../ngram/gramSize.js'
import { sharedFrequencyKernel, type BoundedFrequencyKernel } from '../ngram/kernel.js'
import {
  buildProfile,
  preparedProfile,
  profileOfElements,
  zeroGramSimilarity,
  type NGramProfile,
} from '../ngram/profile.js'

export interface TverskyOptions extends ScorerOptions {
  readonly gramSize?: number | undefined
  readonly alpha?: number | undefined
  readonly beta?: number | undefined
}

interface TverskyParameters {
  readonly alpha: number
  readonly beta: number
}

function validWeight(value: unknown, name: string): number {
  if (value === undefined) return 0.5
  if (typeof value !== 'number') throw new TypeError(`${name} must be a number`)
  if (!(Number.isFinite(value) && value >= 0)) {
    throw new RangeError(`${name} has to be a finite non-negative number`)
  }
  return value
}

function validParameters(rawAlpha: unknown, rawBeta: unknown): TverskyParameters {
  const alpha = validWeight(rawAlpha, 'alpha')
  const beta = validWeight(rawBeta, 'beta')
  if (alpha === 0 && beta === 0) {
    throw new RangeError('alpha and beta must not both be zero')
  }
  return { alpha, beta }
}

function parseParameters(options: Readonly<Record<string, unknown>>): TverskyParameters {
  return validParameters(Reflect.get(options, 'alpha'), Reflect.get(options, 'beta'))
}

function tverskyScore(
  shared: number,
  gramsA: number,
  gramsB: number,
  alpha: number,
  beta: number,
): number {
  // Dividing everything by the largest weight keeps `weight * count` finite
  // for huge coefficients whose true score is still representable; weights at
  // or below 1 leave every operand — and the default Dice arithmetic — as is.
  // The two penalty terms are summed before the numerator joins so that
  // swapping (arguments, weights) only commutes one addition, keeping
  // `T(a, b, α, β)` bit-identical to `T(b, a, β, α)`.
  const scale = Math.max(1, alpha, beta)
  const numerator = shared / scale
  const unmatched =
    (alpha / scale) * (gramsA - shared) + (beta / scale) * (gramsB - shared)
  return numerator / (numerator + unmatched)
}

function similarityBound(
  gramsA: number,
  gramsB: number,
  alpha: number,
  beta: number,
): number {
  return tverskyScore(Math.min(gramsA, gramsB), gramsA, gramsB, alpha, beta)
}

const COUNTER_GRAMS = 512
const COUNTER_GRAM_SIZES: readonly number[] = [2, 3]

function directSimilarity(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  gramSize: number,
  alpha: number,
  beta: number,
  scoreCutoff: number,
): number {
  const gramsA = Math.max(0, a.length - gramSize + 1)
  const gramsB = Math.max(0, b.length - gramSize + 1)
  if (gramsA === 0 || gramsB === 0) {
    const similarity = gramsA === 0 && gramsB === 0 && elementsEqual(a, b) ? 1 : 0
    return similarity >= scoreCutoff ? similarity : 0
  }
  if (similarityBound(gramsA, gramsB, alpha, beta) < scoreCutoff) return 0
  const counted =
    COUNTER_GRAM_SIZES.includes(gramSize) && Math.max(gramsA, gramsB) >= COUNTER_GRAMS
      ? directSharedFrequency(a, b, gramSize)
      : null
  const shared =
    counted ??
    sharedFrequency(profileOfElements(a, gramSize), profileOfElements(b, gramSize))
  const similarity = tverskyScore(shared, gramsA, gramsB, alpha, beta)
  return similarity >= scoreCutoff ? similarity : 0
}

function preparedSimilarity(
  a: NGramProfile,
  shared: BoundedFrequencyKernel,
  b: NGramProfile,
  alpha: number,
  beta: number,
  scoreCutoff: number,
): number {
  const gramsA = a.gramCount
  const gramsB = b.gramCount
  if (gramsA === 0 || gramsB === 0) {
    const similarity = zeroGramSimilarity(a, b)
    return similarity >= scoreCutoff ? similarity : 0
  }
  if (similarityBound(gramsA, gramsB, alpha, beta) < scoreCutoff) return 0
  const similarity = tverskyScore(shared(b, 0), gramsA, gramsB, alpha, beta)
  return similarity >= scoreCutoff ? similarity : 0
}

function tverskySimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: TverskyOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const gramSize = validGramSize(options.gramSize)
  const { alpha, beta } = validParameters(options.alpha, options.beta)
  const scoreCutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normSimCutoff(
    directSimilarity(a, b, gramSize, alpha, beta, scoreCutoff ?? 0),
    scoreCutoff,
  )
}

function tverskyDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: TverskyOptions = {},
): number {
  const gramSize = validGramSize(options.gramSize)
  const { alpha, beta } = validParameters(options.alpha, options.beta)
  const cutoff = options.scoreCutoff
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normDistCutoff(
    1 - directSimilarity(a, b, gramSize, alpha, beta, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

type PreparedTverskyKind = 'distance' | 'similarity'

function prepareTversky(kind: PreparedTverskyKind): PreparationFactory {
  return (options) => {
    const gramSize = parseGramSize(options)
    const { alpha, beta } = parseParameters(options)

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
        const similarity = preparedSimilarity(a, shared, b, alpha, beta, similarityCutoff)
        return kind === 'distance'
          ? normDistCutoff(1 - similarity, rawCutoff)
          : normSimCutoff(similarity, rawCutoff)
      }
    }

    return { prepareQuery, prepareChoice }
  }
}

const tverskyConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  const { alpha, beta } = parseParameters(options)
  return parseGramSize(options) === 2 && alpha === 0.5 && beta === 0.5 ? {} : options
}

const tverskyConfigurationSymmetry: ConfigurationSymmetryResolver = (options) => {
  const { alpha, beta } = parseParameters(options)
  return alpha === beta
}

export const tverskySimilarity: MaybeSequenceMetricImplementation<TverskyOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tverskySimilarity_impl,
    NORMALIZED_SIMILARITY_FLAGS,
    prepareTversky('similarity'),
    {
      configurationCanonicalizer: tverskyConfigurationCanonicalizer,
      configurationSymmetry: tverskyConfigurationSymmetry,
    },
  )
export const tverskyDistance: MaybeSequenceMetricImplementation<TverskyOptions> =
  /* @__PURE__ */ withPreparedFlags(
    tverskyDistance_impl,
    NORMALIZED_DISTANCE_FLAGS,
    prepareTversky('distance'),
    {
      configurationCanonicalizer: tverskyConfigurationCanonicalizer,
      configurationSymmetry: tverskyConfigurationSymmetry,
    },
  )
