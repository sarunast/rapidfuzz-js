import type { PreparedKernel } from '../../core/scoring/compilation.js'
import { directSharedFrequency, sharedFrequency } from '../shared/ngram/compare.js'
import { parseGramSize, validGramSize } from '../shared/ngram/gramSize.js'
import { createDiceIndexBuilder } from '../shared/ngram/inverted/dice.js'
import {
  sharedFrequencyKernel,
  type BoundedFrequencyKernel,
} from '../shared/ngram/kernel.js'
import {
  buildProfile,
  preparedProfile,
  profileOfElements,
  zeroGramSimilarity,
  type NGramProfile,
} from '../shared/ngram/profile.js'
import {
  asSequence,
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

/**
 * `Σ min(a_g, b_g) ≤ min(gramCount(A), gramCount(B))`, so the two gram counts
 * alone cap the similarity. Three operations that reject a short query against
 * a long choice before either trie is walked, which is what makes Dice good at
 * search.
 */
function similarityBound(gramsA: number, gramsB: number): number {
  return (2 * Math.min(gramsA, gramsB)) / (gramsA + gramsB)
}

/**
 * The larger side's gram count from which a transient counter beats building
 * two profiles, and the depths that were measured.
 *
 * Below it the sort a profile pays is cheaper than the `Map` the counter fills.
 * The crossover moves with the **alphabet**, not only the length: at 127 grams
 * bigrams run +22% over a 4-letter alphabet, +5-8% over Latin and -15% over
 * 500 or 2,000 CJK characters, because what the counter pays for is distinct
 * grams. 512 is the first count with no regression in any of those (+0.5% at
 * the symmetric worst case, +45-52% where one side is short), and skewed pairs
 * are where it earns most — a 32-character query against a 4096-character
 * choice sorts 4,000 keys it never needed ordered.
 *
 * Depths 2 and 3 only, listed rather than bounded: `directSharedFrequency`
 * answers depths 1 to 6, and `gramSize <= 3` would route depth 1 along with
 * them. Depth 1 swings from -19% to +57% on the alphabet alone, 4 wins from 128
 * grams, and 5-6 cross near 512 as 3 does — four depths, four answers, so only
 * the two that were measured are named.
 *
 * What this does lose is text with almost no distinct grams — under about 9
 * bigrams or 64 trigrams, where sorting thousands of equal keys is nearly free
 * and a map operation each is not. One repeated gram costs 44-56%, and
 * `bench/ngram.bench.ts`'s `4096 chars, one repeated gram` is what holds that
 * number still. It is a trade rather than an oversight: nothing sees gram
 * diversity before doing the work, it needs *both* sides long — the same
 * repetitive choice against a short query still gains 9-71% — and ordinary
 * text of that length gains 90-120%.
 */
const COUNTER_GRAMS = 512
const COUNTER_GRAM_SIZES: readonly number[] = [2, 3]

/**
 * The gram counts come from the converted elements, never from a string's
 * `length`: `'😀'.length` is 2 and its code-point length is 1, so a UTF-16
 * count overstates the bound's denominator and can reject a candidate that
 * would have qualified. Converting is unavoidable; building the tries is what
 * the bound saves.
 */
function directSimilarity(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  gramSize: number,
  scoreCutoff: number,
): number {
  const gramsA = Math.max(0, a.length - gramSize + 1)
  const gramsB = Math.max(0, b.length - gramSize + 1)
  if (gramsA === 0 || gramsB === 0) {
    // A sequence shorter than `gramSize` has no grams, so the ratio is `0/0`.
    // Two such sequences are as similar as they are equal; against one that
    // does have grams they share none.
    const similarity = gramsA === 0 && gramsB === 0 && elementsEqual(a, b) ? 1 : 0
    return similarity >= scoreCutoff ? similarity : 0
  }
  if (similarityBound(gramsA, gramsB) < scoreCutoff) return 0
  // The counter declines anything it cannot pack — an astral trigram, a mixed
  // domain, an object element — and then the profiles answer, packing a second
  // time before giving up. What that costs depends on where the offending
  // element sits: an early one is free, since the counter abandons at once, and
  // a trailing one costs 6.9-9.2%, having scanned nearly everything first.
  // Against 30-96% won when the same shape does pack.
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

/**
 * A whole number of shared grams the kernel may stop below, so its loop tests
 * integers rather than dividing per gram.
 *
 * Two short of the count the cutoff asks for, and deliberately: the product
 * rounds either way, by far less than one gram but by enough that `Math.ceil`
 * of it can land a step above the true boundary. Slack costs the walk one more
 * group before it gives up; being a step too strict would reject a candidate
 * that scored exactly at the cutoff.
 */
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
  // Anything the kernel stops short of is below the cutoff, and the comparison
  // below turns it into the same `0` the full walk would have produced. An
  // unlimited search never asks for anything, and must not pay a division to
  // be told so.
  const minimumShared = scoreCutoff > 0 ? relaxedShared(denominator, scoreCutoff) : 0
  const similarity = (2 * shared(b, minimumShared)) / denominator
  return similarity >= scoreCutoff ? similarity : 0
}

/**
 * Sørensen-Dice similarity over n-gram frequencies, in `[0, 1]`.
 *
 * Multiset, not set: a gram occurring three times on one side and twice on the
 * other contributes `min(3, 2) = 2` to the shared count, and so four to the
 * numerator. No padding is added at the ends.
 *
 * If the similarity is smaller than `scoreCutoff`, `0` is returned.
 */
function diceSimilarity_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: DiceOptions = {},
): number {
  if (s1 == null || s2 == null) return 0
  const gramSize = validGramSize(options.gramSize)
  const scoreCutoff = options.scoreCutoff
  // `convPair`, not `convSequence` per side: two BMP strings stay strings and
  // pay no conversion at all, and the pair form is what keeps both sides in one
  // element domain — a string against code points compares `'a' !== 97`.
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return normSimCutoff(directSimilarity(a, b, gramSize, scoreCutoff ?? 0), scoreCutoff)
}

/** Sørensen-Dice distance in `[0, 1]`, i.e. `1 - similarity`. */
function diceDistance_impl(
  s1: MaybeSequence,
  s2: MaybeSequence,
  options: DiceOptions = {},
): number {
  const gramSize = validGramSize(options.gramSize)
  const cutoff = options.scoreCutoff
  const [a, b] = convPair(asSequence(s1), asSequence(s2))
  return normDistCutoff(
    1 - directSimilarity(a, b, gramSize, cutoff == null ? 0 : 1 - cutoff),
    cutoff,
  )
}

type PreparedDiceKind = 'distance' | 'similarity'

function prepareDice(kind: PreparedDiceKind): PreparationFactory {
  return (options) => {
    // Once per scorer, so a matcher preparing many queries never reparses it.
    const gramSize = parseGramSize(options)

    const prepareChoice = (choice: Sequence): NGramProfile =>
      buildProfile(choice, gramSize)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = buildProfile(query, gramSize)
      // The query's trie is walked once, here, and never again while the search
      // runs — see `sharedFrequencyKernel`.
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

    // Only the similarity direction: an inverted index accumulates shared gram
    // counts, and a top-k over a distance is a different driver that nothing has
    // measured. The adapter carries `undefined` through, which is what an
    // indexed search refuses on.
    const indexChoices =
      kind === 'similarity' ? () => createDiceIndexBuilder(gramSize) : undefined

    return { prepareQuery, prepareChoice, indexChoices }
  }
}

/**
 * Settle `gramSize` once, when a scorer is compiled, and drop it when it is the
 * default. The adapter reads "is this configured" off the record's keys, so
 * dropping it is what lets `{ gramSize: 2 }` and no configuration at all share
 * one prepared-choice key — while `gramSize: 3` gets its own and refuses a
 * profile built at another depth.
 */
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
