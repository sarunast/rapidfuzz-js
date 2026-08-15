import type { Direction, MaybeSequence, Sequence } from '../types.js'
import type { ChoiceIndexBuilder } from './choiceIndex.js'
import type { OptimumProof } from './optimumProof.js'
import type { AnyBrand } from './preparedChoice.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  /** Returns the actual score when it qualifies; pruning may return only a miss. */
  (choice: unknown, threshold: number | null): number
}

interface Compilation<TDirection extends Direction, TBrand = AnyBrand> {
  readonly direction: TDirection
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly rawScore: (a: Sequence, b: Sequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: (choice: Sequence) => unknown
  readonly prepareOwnedChoice: (choice: Sequence) => unknown
  readonly preparedChoiceKey: object
  readonly indexChoices?: (() => ChoiceIndexBuilder) | undefined
  readonly proveOptimum?: ((prepared: readonly unknown[]) => OptimumProof) | undefined
  readonly preparedChoiceBrand?: TBrand
}

export interface TrustedMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
> extends Compilation<TDirection, TBrand> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

export interface CustomMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
> extends Compilation<TDirection, TBrand> {
  readonly trusted: false
}

export type MetricCompilation<TDirection extends Direction, TBrand = AnyBrand> =
  | TrustedMetricCompilation<TDirection, TBrand>
  | CustomMetricCompilation<TDirection, TBrand>
