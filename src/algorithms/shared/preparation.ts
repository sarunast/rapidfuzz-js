import type { ChoiceIndexBuilder } from '../../core/scoring/choiceIndex.js'
import type { PreparedKernel } from '../../core/scoring/compilation.js'
import type { OptimumProof } from '../../core/scoring/optimumProof.js'
import type { Sequence } from '../../core/types.js'
import {
  distanceCutoffFor,
  distCutoff,
  normalizeDistance,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type MetricScoreKind,
} from './cutoff.js'
import { alignRepresentation, convSequence, scorerSequence } from './sequence.js'

export const PREPARE_SCORER: unique symbol = Symbol('rapidfuzz.prepareScorer')

export type ChoicePreparer = (choice: Sequence) => unknown

export interface MetricPreparation {
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: ChoicePreparer
  readonly indexChoices?: (() => ChoiceIndexBuilder) | undefined
  readonly proveOptimum?: ((prepared: readonly unknown[]) => OptimumProof) | undefined
}

export type PreparationFactory = (
  options: Readonly<Record<string, unknown>>,
) => MetricPreparation

function isPreparedRepresentation(value: unknown): value is ArrayLike<unknown> {
  return (
    typeof value === 'string' ||
    (typeof value === 'object' && value !== null && 'length' in value)
  )
}

export function prepareChoiceSequence(choice: Sequence): ArrayLike<unknown> {
  return scorerSequence(choice)
}

export function preparedChoiceSequence(value: unknown): ArrayLike<unknown> {
  if (!isPreparedRepresentation(value)) {
    throw new TypeError('invalid prepared sequence')
  }

  return value
}

export function prepareMetric(
  kind: MetricScoreKind,
  distance: (
    query: ArrayLike<unknown>,
    choice: ArrayLike<unknown>,
    parsedOptions: unknown,
    distanceCutoff: number,
  ) => number,
  maximum: (query: ArrayLike<unknown>, choice: ArrayLike<unknown>) => number,
  parseOptions: (options: Readonly<Record<string, unknown>>) => unknown = () => null,
): PreparationFactory {
  return (options) => {
    const parsedOptions = parseOptions(options)
    return {
      prepareQuery: (query) => {
        const preparedQuery = scorerSequence(query)
        let convertedQuery: ArrayLike<unknown> | null = null
        return (rawChoice, rawCutoff) => {
          const choice = preparedChoiceSequence(rawChoice)
          const alignedQuery =
            typeof preparedQuery === 'string' && typeof choice !== 'string'
              ? (convertedQuery ??= convSequence(preparedQuery))
              : preparedQuery
          const alignedChoice = alignRepresentation(choice, preparedQuery)
          const max = maximum(alignedQuery, alignedChoice)
          const score = distance(
            alignedQuery,
            alignedChoice,
            parsedOptions,
            distanceCutoffFor(kind, rawCutoff, max),
          )
          switch (kind) {
            case 'distance':
              return distCutoff(score, rawCutoff)
            case 'similarity':
              return simCutoff(max - score, rawCutoff)
            case 'normalizedDistance':
              return normDistCutoff(normalizeDistance(score, max), rawCutoff)
            case 'normalizedSimilarity':
              return normSimCutoff(1 - normalizeDistance(score, max), rawCutoff)
          }
        }
      },
      prepareChoice: prepareChoiceSequence,
    }
  }
}

export type { MetricScoreKind } from './cutoff.js'
