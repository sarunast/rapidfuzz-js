import { alignRepresentation, convSequence, scorerSequence } from '../../sequence.js'
import type { Sequence } from '../../types.js'
import type { CandidateIndexBuilder } from '../candidateIndex.js'
import type { ChoiceIndexBuilder } from '../choiceIndex.js'
import type { PreparedKernel } from '../compilation.js'
import type { OptimumProof } from '../optimumProof.js'
import { distanceCutoffFor, scoreFromDistance, type MetricScoreKind } from './cutoff.js'

export const PREPARE_SCORER: unique symbol = Symbol('rapidfuzz.prepareScorer')

export type ChoicePreparer = (choice: Sequence) => unknown

interface MetricPreparation<TEvidence = never> {
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: ChoicePreparer
  readonly indexChoices?: (() => ChoiceIndexBuilder) | undefined
  readonly candidateChoices?: (() => CandidateIndexBuilder) | undefined
  readonly proveOptimum?: ((prepared: readonly unknown[]) => OptimumProof) | undefined
  readonly explain?: ((first: Sequence, second: Sequence) => TEvidence) | undefined
}

export type PreparationFactory<TEvidence = never> = (
  options: Readonly<Record<string, unknown>>,
) => MetricPreparation<TEvidence>

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
          return scoreFromDistance(kind, score, max, rawCutoff)
        }
      },
      prepareChoice: prepareChoiceSequence,
    }
  }
}

export type { MetricScoreKind } from './cutoff.js'
