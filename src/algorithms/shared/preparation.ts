import type { Sequence } from '../../core/types.js'
import {
  distanceCutoffFor,
  distCutoff,
  normalize,
  normSimCutoff,
  type PreparedMetricKind,
} from './cutoff.js'
import { alignRepresentation, scorerSequence } from './sequence.js'

export const PREPARE_CHOICE: unique symbol = Symbol('rapidfuzz.prepareChoice')
export const PREPARE_SCORER: unique symbol = Symbol('rapidfuzz.prepareScorer')

const PREPARED_SEQUENCE = Symbol('rapidfuzz.preparedSequence')

interface PreparedSequence {
  readonly [PREPARED_SEQUENCE]: true
  readonly value: ArrayLike<unknown>
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

export function prepareScorerChoice(choice: Sequence): unknown {
  const prepared: PreparedSequence = {
    [PREPARED_SEQUENCE]: true,
    value: scorerSequence(choice),
  }
  return prepared
}

export function preparedScorerSequence(value: unknown): ArrayLike<unknown> {
  // oxlint-disable-next-line typescript/consistent-type-assertions -- private preparation protocol owns this value
  return (value as PreparedSequence).value
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
      return kind === 'distance'
        ? distCutoff(score, rawCutoff)
        : normSimCutoff(1 - normalize(score, max), rawCutoff)
    }
  }
  return withChoicePreparer(prepare, prepareScorerChoice)
}

export type { PreparedMetricKind } from './cutoff.js'
