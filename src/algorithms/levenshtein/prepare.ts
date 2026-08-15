import type { PreparedKernel } from '../../core/scoring/compilation.js'
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

const MAX_BAND_BUDGET = 15

type PreparedLevenshteinKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

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

function preparedBandWorthwhile(
  queryLength: number,
  choiceLength: number,
  budget: number,
): boolean {
  if (!Number.isFinite(budget) || budget < 4) return false

  if (Math.abs(queryLength - choiceLength) > budget) return false

  return (
    budget <= queryLength &&
    budget <= choiceLength &&
    choiceLength >= queryLength - budget
  )
}

const AFFIX_TRIM_WORDS = 4

function worthTrimming(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  const words = wordCount(Math.min(a.length, b.length))
  return words >= AFFIX_TRIM_WORDS && sharesWideAffix(a, b)
}

export function prepareLevenshtein(kind: PreparedLevenshteinKind): PreparationFactory {
  return (options) => {
    const weights = parseWeights(Reflect.get(options, 'weights'))
    const [insert, delete_, replace] = weights
    const uniform = insert === 1 && delete_ === 1 && replace === 1
    const integral = integralWeights(weights)

    const prepareQuery = (query: Sequence): PreparedKernel => {
      const a = scorerSequence(query)
      let pattern: PatternMask | null = null

      const preparedDistance = (b: ArrayLike<unknown>, cutoff: number): number => {
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
