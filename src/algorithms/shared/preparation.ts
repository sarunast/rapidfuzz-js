import type { Sequence } from '../../core/types.js'
import {
  distanceCutoffFor,
  distCutoff,
  normalize,
  normDistCutoff,
  normSimCutoff,
  simCutoff,
  type PreparedMetricKind,
} from './cutoff.js'
import { alignRepresentation, scorerSequence } from './sequence.js'

export const PREPARE_CHOICE: unique symbol = Symbol('rapidfuzz.prepareChoice')
export const PREPARE_SCORER: unique symbol = Symbol('rapidfuzz.prepareScorer')

class PreparedSequence {
  constructor(readonly value: ArrayLike<unknown>) {}
}

export type ChoicePreparer = (choice: Sequence) => unknown

export interface PreparedScore {
  (choice: unknown, scoreCutoff: number | null): number
}

export interface PrepareScorer {
  (query: Sequence, options: Readonly<Record<string, unknown>>): PreparedScore
  [PREPARE_CHOICE]?: ChoicePreparer
}

export interface PreparedScorerFactory extends PrepareScorer {
  [PREPARE_CHOICE]: ChoicePreparer
}

export function withChoicePreparer(
  prepare: PrepareScorer,
  choicePreparer: ChoicePreparer,
): PreparedScorerFactory {
  prepare[PREPARE_CHOICE] = choicePreparer
  return Object.assign(prepare, { [PREPARE_CHOICE]: choicePreparer })
}

export function prepareScorerChoice(choice: Sequence): PreparedSequence {
  return new PreparedSequence(scorerSequence(choice))
}

export function preparedScorerSequence(value: unknown): ArrayLike<unknown> {
  if (!(value instanceof PreparedSequence)) {
    throw new TypeError('invalid prepared sequence')
  }

  return value.value
}

export function prepareMetric(
  kind: PreparedMetricKind,
  distance: (
    query: ArrayLike<unknown>,
    choice: ArrayLike<unknown>,
    parsedOptions: unknown,
    distanceCutoff: number,
  ) => number,
  maximum: (query: ArrayLike<unknown>, choice: ArrayLike<unknown>) => number,
  parseOptions: (options: Readonly<Record<string, unknown>>) => unknown = () => null,
): PreparedScorerFactory {
  const prepare: PrepareScorer = (query, options) => {
    const preparedQuery = preparedScorerSequence(prepareScorerChoice(query))
    const parsedOptions = parseOptions(options)
    return (rawChoice, rawCutoff) => {
      const choice = preparedScorerSequence(rawChoice)
      const alignedQuery = alignRepresentation(preparedQuery, choice)
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
          return normDistCutoff(normalize(score, max), rawCutoff)
        case 'normalizedSimilarity':
          return normSimCutoff(1 - normalize(score, max), rawCutoff)
      }
    }
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

export type { PreparedMetricKind } from './cutoff.js'
