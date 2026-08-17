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
import type {
  ChoicePreparer,
  PreparationFactory,
} from '#core/scoring/builtIn/preparation.js'
import type { ChoiceIndexBuilder } from '#core/scoring/choiceIndex.js'
import type { PreparedKernel } from '#core/scoring/compilation.js'
import {
  validateSequence,
  convPair,
  convSequence,
  elementsEqual,
} from '#core/sequence.js'
import type { MaybeSequence, Sequence } from '#core/types.js'

import { directSharedFrequency, sharedFrequency } from '../ngram/compare.js'
import { parseGramSize, validGramSize } from '../ngram/gramSize.js'
import { createDiceIndexBuilder } from '../ngram/inverted/dice.js'
import { createTverskyIndexBuilder } from '../ngram/inverted/tversky.js'
import { createWeightedTverskyIndexBuilder } from '../ngram/inverted/weightedTversky.js'
import { sharedFrequencyKernel, type BoundedFrequencyKernel } from '../ngram/kernel.js'
import {
  buildProfile,
  preparedProfile,
  profileOfElements,
  zeroGramSimilarity,
  type NGramProfile,
} from '../ngram/profile.js'
import {
  compileElementWeights,
  CompiledElementWeights,
  preparedWeightedProfile,
  weightedComponents,
  weightedProfile,
  weightedQueryGroups,
  zeroMassSimilarity,
  type WeightedProfile,
  type WeightedQueryGroups,
} from '../ngram/weightedProfile.js'
import { weightedTverskyScore } from '../ngram/weightedTverskyScore.js'
import {
  CompiledElementSimilarity,
  compileElementSimilarity,
  effectiveElementSimilarity,
  ElementKernels,
  type TverskyElementSimilarity,
} from './elementSimilarity.js'
import { tverskyExplainer, type TverskyEvidence } from './evidence.js'
import { canonicalElements, occurrencesOf, type Occurrence } from './occurrences.js'
import { tverskyScore } from './score.js'
import {
  preparedSoftChoice,
  softComponentsOf,
  softQueryOf,
  SoftTverskyChoice,
  softTablesOf,
  type SoftQuery,
} from './soft.js'

export interface TverskyOptions extends ScorerOptions {
  readonly gramSize?: number | undefined
  readonly alpha?: number | undefined
  readonly beta?: number | undefined
  /**
   * A caller's map, or the table a configured scorer compiled it into once —
   * the canonicalizer replaces one with the other, so a compiled scorer never
   * recompiles and never sees a later mutation.
   */
  readonly elementWeights?:
    | ReadonlyMap<unknown, number>
    | CompiledElementWeights
    | undefined
  readonly defaultElementWeight?: number | undefined
  /**
   * A caller's option, or the compiled form the canonicalizer replaced it with
   * — the same public-input/internal-compiled pattern the weights follow.
   */
  readonly elementSimilarity?:
    | TverskyElementSimilarity
    | CompiledElementSimilarity
    | undefined
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

function weightsRequested(rawWeights: unknown, rawDefault: unknown): boolean {
  return rawWeights !== undefined || rawDefault !== undefined
}

/**
 * Element weights ask for exact element overlap, which is what `gramSize: 1`
 * is: a shingle of several elements has no single weight to carry, and no
 * combining rule over its elements is more right than another.
 */
function compileWeights(
  rawWeights: unknown,
  rawDefault: unknown,
  gramSize: number,
): CompiledElementWeights {
  if (gramSize !== 1) {
    throw new RangeError('element weights are only defined at gramSize 1')
  }
  return rawWeights instanceof CompiledElementWeights
    ? rawWeights
    : compileElementWeights(rawWeights, rawDefault)
}

/**
 * The compiled table, or `null` where the weights price nothing — no weight key
 * at all, or every element weighing the same positive amount, which is ordinary
 * unigram Tversky and is answered by the unweighted engines.
 *
 * Every entry point that reads the configuration asks this one question, so a
 * uniform table cannot take the weighted path through one of them and the
 * unweighted path through another: the two agree on the real number and need not
 * agree on its last bit.
 */
function effectiveWeights(
  rawWeights: unknown,
  rawDefault: unknown,
  gramSize: number,
): CompiledElementWeights | null {
  if (!weightsRequested(rawWeights, rawDefault)) return null
  const weights = compileWeights(rawWeights, rawDefault, gramSize)
  return weights.uniformPositive ? null : weights
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

// Three components at a time, and never two calls deep: the weighted paths read
// them back before anything else can run.
const WEIGHTED_PARTS = /* @__PURE__ */ new Float64Array(3)

function weightedSimilarity(
  query: WeightedQueryGroups,
  choice: WeightedProfile,
  weights: CompiledElementWeights,
  alpha: number,
  beta: number,
): number {
  if (query.groupIds.length === 0 || choice.groupIds.length === 0) {
    return zeroMassSimilarity(query, choice)
  }
  weightedComponents(query, choice, weights, WEIGHTED_PARTS)
  return weightedTverskyScore(
    WEIGHTED_PARTS[0],
    WEIGHTED_PARTS[1],
    WEIGHTED_PARTS[2],
    alpha,
    beta,
  )
}

function directWeightedSimilarity(
  a: Sequence,
  b: Sequence,
  weights: CompiledElementWeights,
  alpha: number,
  beta: number,
  scoreCutoff: number,
): number {
  // `convSequence` on both sides rather than `convPair`, which keeps two
  // surrogate-free strings as strings: weight keys are canonical elements, so
  // `'a'` has to arrive as the `97` the table was built with.
  const similarity = weightedSimilarity(
    weightedQueryGroups(convSequence(a), weights),
    weightedProfile(convSequence(b), weights),
    weights,
    alpha,
    beta,
  )
  return similarity >= scoreCutoff ? similarity : 0
}

/**
 * Exact unigram overlap completed by fuzzy matching of what it left over.
 *
 * A soft configuration gets its own engine because the packed profiles and
 * weight groups the exact paths ride count grams without saying *which*
 * occurrences are left over, which is exactly what the reservation needs.
 *
 * The exact answer is computed first and returned verbatim whenever the fuzzy
 * phase adds nothing, so a configuration whose threshold nothing reaches is
 * bit-identical to the same configuration without one. The two fold their
 * penalties in different orders — per element here, per weight group there —
 * and short-circuiting is what keeps that from being observable.
 */
function softSimilarity(
  query: SoftQuery,
  second: readonly Occurrence[],
  soft: CompiledElementSimilarity,
  alpha: number,
  beta: number,
  kernels: ElementKernels | null,
): number {
  const first = query.occurrences
  const weighted = query.weighted
  const tables = softTablesOf(query.table, second)
  if (weighted === null) {
    if (first.length === 0 || second.length === 0) {
      return first.length === second.length ? 1 : 0
    }
    const shared = tables.overlap.sharedCount
    const components = softComponentsOf(tables, soft, shared, kernels)
    return components === null
      ? tverskyScore(shared, first.length, second.length, alpha, beta)
      : weightedTverskyScore(
          components.shared,
          components.firstOnly,
          components.secondOnly,
          alpha,
          beta,
        )
  }
  const groups = weighted.groups
  const choice = weightedProfile(canonicalElements(second), weighted.weights)
  if (groups.groupIds.length === 0 || choice.groupIds.length === 0) {
    return zeroMassSimilarity(groups, choice)
  }
  // Its own array rather than the module scratch: the element scorer below can
  // be another weighted Tversky, which would write over a shared one.
  const parts = new Float64Array(3)
  weightedComponents(groups, choice, weighted.weights, parts)
  const components = softComponentsOf(tables, soft, parts[0], kernels)
  return components === null
    ? weightedTverskyScore(parts[0], parts[1], parts[2], alpha, beta)
    : weightedTverskyScore(
        components.shared,
        components.firstOnly,
        components.secondOnly,
        alpha,
        beta,
      )
}

function directSoftSimilarity(
  a: Sequence,
  b: Sequence,
  weights: CompiledElementWeights | null,
  soft: CompiledElementSimilarity,
  alpha: number,
  beta: number,
  scoreCutoff: number,
): number {
  const similarity = softSimilarity(
    softQueryOf(a, weights),
    occurrencesOf(b, weights),
    soft,
    alpha,
    beta,
    null,
  )
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
  // Deliberately unbounded: inverting the threshold into a minimum shared
  // count is not float-safe here. Tiny weights round the score up to the
  // brink of 1 while the real-number inversion still demands more shared
  // grams than a qualifying candidate has, and the bounded walk would then
  // under-count it into a false rejection.
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
  const weights = effectiveWeights(
    options.elementWeights,
    options.defaultElementWeight,
    gramSize,
  )
  const soft = effectiveElementSimilarity(options.elementSimilarity, gramSize)
  const scoreCutoff = options.scoreCutoff
  if (soft !== null) {
    return normSimCutoff(
      directSoftSimilarity(
        validateSequence(s1),
        validateSequence(s2),
        weights,
        soft,
        alpha,
        beta,
        scoreCutoff ?? 0,
      ),
      scoreCutoff,
    )
  }
  if (weights !== null) {
    return normSimCutoff(
      directWeightedSimilarity(
        validateSequence(s1),
        validateSequence(s2),
        weights,
        alpha,
        beta,
        scoreCutoff ?? 0,
      ),
      scoreCutoff,
    )
  }
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
  const weights = effectiveWeights(
    options.elementWeights,
    options.defaultElementWeight,
    gramSize,
  )
  const soft = effectiveElementSimilarity(options.elementSimilarity, gramSize)
  const cutoff = options.scoreCutoff
  const similarityCutoff = cutoff == null ? 0 : 1 - cutoff
  if (soft !== null) {
    return normDistCutoff(
      1 -
        directSoftSimilarity(
          validateSequence(s1),
          validateSequence(s2),
          weights,
          soft,
          alpha,
          beta,
          similarityCutoff,
        ),
      cutoff,
    )
  }
  if (weights !== null) {
    return normDistCutoff(
      1 -
        directWeightedSimilarity(
          validateSequence(s1),
          validateSequence(s2),
          weights,
          alpha,
          beta,
          similarityCutoff,
        ),
      cutoff,
    )
  }
  const [a, b] = convPair(validateSequence(s1), validateSequence(s2))
  return normDistCutoff(
    1 - directSimilarity(a, b, gramSize, alpha, beta, similarityCutoff),
    cutoff,
  )
}

type PreparedTverskyKind = 'distance' | 'similarity'

function similarityCutoffFor(
  kind: PreparedTverskyKind,
  rawCutoff: number | null,
): number {
  if (kind === 'distance') return rawCutoff === null ? 0 : 1 - rawCutoff
  return rawCutoff ?? 0
}

function preparedResult(
  kind: PreparedTverskyKind,
  similarity: number,
  rawCutoff: number | null,
): number {
  return kind === 'distance'
    ? normDistCutoff(1 - similarity, rawCutoff)
    : normSimCutoff(similarity, rawCutoff)
}

/**
 * The weighted preparation, chosen once when the scorer compiles so that no
 * weight test reaches an unweighted profile or overlap loop.
 */
function prepareWeightedTversky(
  kind: PreparedTverskyKind,
  weights: CompiledElementWeights,
  alpha: number,
  beta: number,
): {
  prepareQuery: (query: Sequence) => PreparedKernel
  prepareChoice: ChoicePreparer
  indexChoices?: (() => ChoiceIndexBuilder) | undefined
  explain: (first: Sequence, second: Sequence) => TverskyEvidence
} {
  return {
    // Weights are only accepted at `gramSize: 1`, so a weighted scorer always
    // explains.
    explain: tverskyExplainer(kind, weights, alpha, beta),
    // Weighted overlap gets its own index at every weight pair, the default
    // included: Dice's knows nothing about element weights.
    indexChoices:
      kind === 'similarity'
        ? () =>
            createWeightedTverskyIndexBuilder(
              alpha,
              beta,
              weights.groupWeights,
              weights.groupOf,
              weights.defaultGroup,
            )
        : undefined,
    prepareChoice: (choice: Sequence): WeightedProfile =>
      weightedProfile(convSequence(choice), weights),
    prepareQuery: (query: Sequence): PreparedKernel => {
      const groups = weightedQueryGroups(convSequence(query), weights)
      return (rawChoice, rawCutoff) => {
        const choice = preparedWeightedProfile(rawChoice)
        const similarity = weightedSimilarity(groups, choice, weights, alpha, beta)
        const cutoff = similarityCutoffFor(kind, rawCutoff)
        return preparedResult(kind, similarity >= cutoff ? similarity : 0, rawCutoff)
      }
    },
  }
}

/**
 * The soft preparation. It offers no `indexChoices` in either direction: the
 * inverted indexes score exact overlap, so an indexed matcher built on one
 * would quietly disagree with the exhaustive scorer it is meant to reproduce.
 */
function prepareSoftTversky(
  kind: PreparedTverskyKind,
  weights: CompiledElementWeights | null,
  soft: CompiledElementSimilarity,
  alpha: number,
  beta: number,
): {
  prepareQuery: (query: Sequence) => PreparedKernel
  prepareChoice: ChoicePreparer
  indexChoices?: (() => ChoiceIndexBuilder) | undefined
  explain: (first: Sequence, second: Sequence) => TverskyEvidence
} {
  return {
    // Element similarity is only accepted at `gramSize: 1`, so a soft scorer
    // always explains.
    explain: tverskyExplainer(kind, weights, alpha, beta, soft),
    indexChoices: undefined,
    prepareChoice: (choice: Sequence): SoftTverskyChoice =>
      new SoftTverskyChoice(occurrencesOf(choice, weights)),
    prepareQuery: (query: Sequence): PreparedKernel => {
      // Everything the query side derives, done once and met with every
      // candidate — the one place in soft scoring that has a scan to amortize
      // over. The element scorer's own preparation is held beside it, and is
      // the one part of it that waits to be earned.
      const first = softQueryOf(query, weights)
      const kernels = new ElementKernels(soft)
      return (rawChoice, rawCutoff) => {
        const second = preparedSoftChoice(rawChoice).occurrences
        const similarity = softSimilarity(first, second, soft, alpha, beta, kernels)
        const cutoff = similarityCutoffFor(kind, rawCutoff)
        return preparedResult(kind, similarity >= cutoff ? similarity : 0, rawCutoff)
      }
    },
  }
}

function prepareTversky(kind: PreparedTverskyKind): PreparationFactory<TverskyEvidence> {
  return (options) => {
    const gramSize = parseGramSize(options)
    const { alpha, beta } = parseParameters(options)
    const weights = effectiveWeights(
      Reflect.get(options, 'elementWeights'),
      Reflect.get(options, 'defaultElementWeight'),
      gramSize,
    )
    const soft = effectiveElementSimilarity(
      Reflect.get(options, 'elementSimilarity'),
      gramSize,
    )
    if (soft !== null) return prepareSoftTversky(kind, weights, soft, alpha, beta)
    if (weights !== null) return prepareWeightedTversky(kind, weights, alpha, beta)

    const prepareChoice = (choice: Sequence): NGramProfile =>
      buildProfile(choice, gramSize)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = buildProfile(query, gramSize)
      const shared = sharedFrequencyKernel(a)

      return (rawChoice, rawCutoff) => {
        const b = preparedProfile(rawChoice)
        const similarity = preparedSimilarity(
          a,
          shared,
          b,
          alpha,
          beta,
          similarityCutoffFor(kind, rawCutoff),
        )
        return preparedResult(kind, similarity, rawCutoff)
      }
    }

    // Default weights are exactly Dice — the halved penalty sum is `(A+B)/2`
    // and power-of-two scaling never changes IEEE rounding — so they share
    // Dice's index, hot loop and arithmetic.
    const indexChoices =
      kind === 'similarity'
        ? alpha === 0.5 && beta === 0.5
          ? () => createDiceIndexBuilder(gramSize)
          : () => createTverskyIndexBuilder(gramSize, alpha, beta)
        : undefined

    // Only exact element overlap has occurrences to explain: a shingle of
    // several elements is not one thing a caller named.
    const explain = gramSize === 1 ? tverskyExplainer(kind, null, alpha, beta) : undefined

    return { prepareQuery, prepareChoice, indexChoices, explain }
  }
}

function withoutWeights(
  options: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> {
  const rest: Record<string, unknown> = { ...options }
  Reflect.deleteProperty(rest, 'elementWeights')
  Reflect.deleteProperty(rest, 'defaultElementWeight')
  return rest
}

/**
 * Compiles the element weights once, here, because this runs when the scorer
 * compiles and its answer is the configuration every later call reads: a
 * configured scorer therefore cannot observe a mutation of the caller's map, and
 * pays for validation once rather than per pair.
 *
 * A uniform positive weighting is dropped outright, since it multiplies all
 * three components by one constant and cancels: the scorer that follows is
 * plain unigram Tversky, over Tversky's own index and profiles, rather than a
 * weighted representation that measured 4.3x the index and 3.1x the direct score
 * for nothing. `defaultElementWeight: 0` is deliberately not uniform — it means
 * the ignored-element rules, which are their own semantics.
 *
 * The default-configuration collapse has to stay behind all of that — a
 * genuinely weighted scorer that canonicalized to `{}` would silently score
 * unweighted.
 */
const tverskyConfigurationCanonicalizer: ConfigurationCanonicalizer = (options) => {
  const { alpha, beta } = parseParameters(options)
  const gramSize = parseGramSize(options)
  // Compiled before either return below, so a weight branch cannot carry an
  // unvalidated element similarity forward.
  const rawSoft = Reflect.get(options, 'elementSimilarity')
  const base =
    rawSoft === undefined
      ? options
      : { ...options, elementSimilarity: compileElementSimilarity(rawSoft, gramSize) }
  const rawWeights = Reflect.get(options, 'elementWeights')
  const rawDefault = Reflect.get(options, 'defaultElementWeight')
  if (weightsRequested(rawWeights, rawDefault)) {
    const weights = effectiveWeights(rawWeights, rawDefault, gramSize)
    return weights === null ? withoutWeights(base) : { ...base, elementWeights: weights }
  }
  // No `elementSimilarity` guard on the collapse: it demands `gramSize: 2`,
  // which compiling an element similarity has already refused.
  return gramSize === 2 && alpha === 0.5 && beta === 0.5 ? {} : base
}

const tverskyConfigurationSymmetry: ConfigurationSymmetryResolver = (options) => {
  const { alpha, beta } = parseParameters(options)
  // A soft score is symmetric as a real number where alpha equals beta, but the
  // solver's tie-break and the element-order fold are not transpose invariant,
  // and `scoreMatrix` mirrors a symmetric scorer's cells rather than scoring
  // them — a wrong `true` would be wrong numbers, not wasted work.
  return alpha === beta && Reflect.get(options, 'elementSimilarity') === undefined
}

export const tverskySimilarity: MaybeSequenceMetricImplementation<
  TverskyOptions,
  TverskyEvidence
> = /* @__PURE__ */ withPreparedFlags(
  tverskySimilarity_impl,
  NORMALIZED_SIMILARITY_FLAGS,
  prepareTversky('similarity'),
  {
    configurationCanonicalizer: tverskyConfigurationCanonicalizer,
    configurationSymmetry: tverskyConfigurationSymmetry,
  },
)
export const tverskyDistance: MaybeSequenceMetricImplementation<
  TverskyOptions,
  TverskyEvidence
> = /* @__PURE__ */ withPreparedFlags(
  tverskyDistance_impl,
  NORMALIZED_DISTANCE_FLAGS,
  prepareTversky('distance'),
  {
    configurationCanonicalizer: tverskyConfigurationCanonicalizer,
    configurationSymmetry: tverskyConfigurationSymmetry,
  },
)
