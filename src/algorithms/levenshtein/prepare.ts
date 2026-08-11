import type { PreparedKernel } from '../../core/protocol.js'
import type { Sequence } from '../../core/types.js'
import { sharesWideAffix } from '../shared/affix.js'
import { wordCount } from '../shared/bitmask/blockMasks.js'
import { preparePattern, type PatternMask } from '../shared/bitmask/pattern.js'
import {
  alignRepresentation,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  prepareChoiceSequence,
  preparedChoiceSequence,
  scorerSequence,
  type PreparationFactory,
} from '../shared/scorerSupport.js'
import {
  distance_,
  integralWeights,
  levenshteinRawCutoff,
  levenshteinSimilarityCutoff,
  maximum,
  parseWeights,
  rawBound,
} from './internal/engine.js'
import { levenshteinPrepared, levenshteinSmallBand } from './internal/uniform.js'

/**
 * Widest budget {@link levenshteinSmallBand} can band inside one word.
 *
 * Its diagonal band is `2 * budget + 1` wide and it holds the band in a single
 * 32-bit word, so anything past 15 would have the part that does not fit
 * dropped and come back too large.
 */
const MAX_BAND_BUDGET = 15

type PreparedLevenshteinKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

/**
 * Whether the held-pattern kernel beats rebuilding the query's masks per choice.
 *
 * Four things have to hold, and each of them is what makes the prepared result
 * identical to the unprepared one rather than merely close.
 *
 * The weights must be uniform, since only then does {@link distance_} reach the
 * plain Myers kernel rather than the LCS or generic ones.
 *
 * The budget and the hint must both be wide enough that {@link levenshteinUniform}
 * would have run its unbounded kernel. A banded run reports `budget + 1` for
 * anything out of reach, which the exact kernel below never does; narrower
 * budgets are served by {@link preparedBandWorthwhile} instead.
 *
 * The query must be the side {@link levenshteinUniform} would itself have made
 * the pattern. Masks for the other side would have to be rebuilt for every
 * choice anyway, which is the cost being avoided.
 */
function preparedDistanceWorthwhile(
  uniform: boolean,
  queryLength: number,
  choiceLength: number,
  scoreCutoff: number,
): boolean {
  if (!uniform || queryLength === 0 || choiceLength === 0) return false

  const longest = Math.max(queryLength, choiceLength)
  if (Math.floor(scoreCutoff) < longest) return false

  const queryWords = wordCount(queryLength)
  const choiceWords = wordCount(choiceLength)
  return queryWords * choiceLength <= choiceWords * queryLength
}

/**
 * Whether the held pattern can serve a budgeted score through the small band.
 *
 * `extract` tightens its cutoff as the heap fills, so nearly every choice it
 * scores arrives with a budget — which is what kept it on the unprepared path
 * while `scoreMatrix` moved onto the held masks.
 *
 * {@link levenshteinSmallBand} reads its masks through `shiftedPatternMatches`,
 * which windows a pattern of any width, so the whole-query masks serve it
 * unchanged. What it will not do is fit a band wider than a word: the budget
 * has to satisfy `2 * budget + 1 <= 32`, or the kernel drops the part that does
 * not fit and answers too large. {@link MAX_BAND_BUDGET} is that bound, and the
 * caller applies it — it still covers what `extract` asks for, since a heap of
 * the best few over short text settles on a budget well inside it.
 *
 * The remaining bounds are the kernel's other two preconditions, plus a length
 * test that is only an early out.
 *
 * Budgets under four are left alone: `levenshteinMbleven` answers those by
 * comparing elements directly, and builds no masks to save.
 */
function preparedBandWorthwhile(
  queryLength: number,
  choiceLength: number,
  budget: number,
): boolean {
  if (!Number.isFinite(budget) || budget < 4) return false

  // Out of reach on length alone: `distance_` returns `budget + 1` for this
  // without touching a mask, so there is nothing to save.
  if (Math.abs(queryLength - choiceLength) > budget) return false

  return (
    budget <= queryLength &&
    budget <= choiceLength &&
    choiceLength >= queryLength - budget
  )
}

/**
 * Fewest words a pattern has to span before trimming an affix beats holding
 * its masks.
 *
 * The held pattern's whole advantage is the mask build it does not repeat per
 * choice; trimming's is the elements the kernel never reaches, and each of
 * those is worth one step *per word*. So the two scale differently, and below
 * four words trimming loses even when the affix is nearly the whole input: at
 * two words a pair sharing 56 of 64 elements measured 0.83x, and at three words
 * an affix of 84 in 96 bought 1.13x against 0.92x on the near-copies beside it.
 * Four words is where the win clears what the probe costs — 1.5x there, 2.6x at
 * eight words and 9.2x at sixteen.
 */
const AFFIX_TRIM_WORDS = 4

/**
 * Whether the pair is better served by trimming its common affix than by the
 * held pattern the length gate has just approved.
 *
 * {@link preparedDistanceWorthwhile} answers from lengths, and lengths cannot
 * see an affix: a query and a choice of 512 elements that agree on 480 of them
 * look exactly like two unrelated ones. The held pattern reads all 512 either
 * way, where the unprepared kernel trims first and scores 32.
 *
 * The same relaxation the LCS metrics take, in the opposite direction — there
 * the probe lets a pair *onto* the held masks that a length gate refused, here
 * it takes one *off* them. That is why it asks {@link sharesWideAffix} for a
 * quarter rather than {@link sharesAffix}'s eighth, and why a pattern narrower
 * than {@link AFFIX_TRIM_WORDS} is not asked at all.
 */
function worthTrimming(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  const words = wordCount(Math.min(a.length, b.length))
  return words >= AFFIX_TRIM_WORDS && sharesWideAffix(a, b)
}

export function prepareLevenshtein(kind: PreparedLevenshteinKind): PreparationFactory {
  return (options) => {
    // Once per scorer, so a matcher preparing many queries never reparses them.
    const weights = parseWeights(Reflect.get(options, 'weights'))
    const [insert, delete_, replace] = weights
    const uniform = insert === 1 && delete_ === 1 && replace === 1
    const integral = integralWeights(weights)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = scorerSequence(query)
      let pattern: PatternMask | null = null

      const preparedDistance = (b: ArrayLike<unknown>, cutoff: number): number => {
        // A scorer with no cutoff runs with `cutoff` at `MAX_SAFE_INTEGER`, which
        // is what `scoreMatrix` does, and this comparison keeps it from paying for the
        // rest of the band test. It is also the only place the band's width bound
        // is applied, so `preparedBandWorthwhile` can take the budget as given.
        if (cutoff < MAX_BAND_BUDGET + 1 && uniform && a.length > 0 && b.length > 0) {
          const budget = Math.floor(cutoff)
          if (preparedBandWorthwhile(a.length, b.length, budget)) {
            pattern ??= preparePattern(a, 0, a.length)
            return levenshteinSmallBand(pattern, a.length, b, 0, b.length, budget)
          }
        }
        if (
          preparedDistanceWorthwhile(uniform, a.length, b.length, cutoff) &&
          !worthTrimming(a, b)
        ) {
          pattern ??= preparePattern(a, 0, a.length)
          return levenshteinPrepared(pattern, b, 0, b.length)
        }
        // The unprepared kernel trims a common affix, which compares the two
        // sequences elementwise, so they have to agree on how a character is
        // spelled. The held pattern above reads either representation.
        return distance_(
          alignRepresentation(a, b),
          alignRepresentation(b, a),
          weights,
          cutoff,
          cutoff,
        )
      }

      const score: PreparedKernel = (rawChoice, rawCutoff) => {
        const b = preparedChoiceSequence(rawChoice)
        // Lengths are all `maximum` reads, and aligning cannot change them, so it
        // is left to the unprepared path in `preparedDistance` that needs it.
        const max = maximum(a, b, weights)
        switch (kind) {
          case 'distance': {
            const cutoff = levenshteinRawCutoff(rawCutoff, integral)
            const bound = cutoff ?? Number.MAX_SAFE_INTEGER
            const distance = preparedDistance(b, bound)
            return cutoff === null || distance <= cutoff ? distance : cutoff + 1
          }
          case 'similarity': {
            const cutoff = levenshteinSimilarityCutoff(rawCutoff, integral)
            const bound =
              cutoff === null ? Number.MAX_SAFE_INTEGER : rawBound(max - cutoff, integral)
            return simCutoff(max - preparedDistance(b, bound), cutoff)
          }
          case 'normalizedDistance': {
            const cutoff =
              rawCutoff === null
                ? Number.MAX_SAFE_INTEGER
                : rawBound(rawCutoff * max, integral)
            return normDistCutoff(
              normalizeDistance(preparedDistance(b, cutoff), max),
              rawCutoff,
            )
          }
          case 'normalizedSimilarity': {
            const cutoff =
              rawCutoff === null
                ? Number.MAX_SAFE_INTEGER
                : rawBound((1 - rawCutoff) * max, integral)
            return normSimCutoff(
              1 - normalizeDistance(preparedDistance(b, cutoff), max),
              rawCutoff,
            )
          }
        }
      }
      return score
    }

    return { prepareQuery, prepareChoice: prepareChoiceSequence }
  }
}
