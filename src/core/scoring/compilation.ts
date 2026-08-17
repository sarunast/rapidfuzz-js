import type { Direction, MaybeSequence, Sequence } from '../types.js'
import type { ChoiceIndexBuilder } from './choiceIndex.js'
import type { OptimumProof } from './optimumProof.js'
import type { AnyBrand } from './preparedChoice.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  /** Returns the actual score when it qualifies; pruning may return only a miss. */
  (choice: unknown, threshold: number | null): number
}

interface Compilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
  TExplains extends object = never,
  TEvidence = never,
> {
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
  readonly explain?: ((first: Sequence, second: Sequence) => TEvidence) | undefined
  readonly preparedChoiceBrand?: TBrand
  readonly explainedConfiguration?: TExplains
}

export interface TrustedMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
  TExplains extends object = never,
  TEvidence = never,
> extends Compilation<TDirection, TBrand, TExplains, TEvidence> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

export interface CustomMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
  TExplains extends object = never,
  TEvidence = never,
> extends Compilation<TDirection, TBrand, TExplains, TEvidence> {
  readonly trusted: false
}

export type MetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
  TExplains extends object = never,
  TEvidence = never,
> =
  | TrustedMetricCompilation<TDirection, TBrand, TExplains, TEvidence>
  | CustomMetricCompilation<TDirection, TBrand, TExplains, TEvidence>

/**
 * A compilation read back without its capability types.
 *
 * Every capability-carrying compilation is assignable to it, because both
 * erased parameters occur only in covariant positions. This is one of the two
 * sanctioned places a capability is dropped: everything that merely *runs* a
 * compilation takes this, and everything that *builds* one keeps the real
 * parameters. Never spell the erasure inline as
 * `MetricCompilation<…, object, unknown>` — the whole point is that it is
 * greppable.
 */
export type AnyMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
> = MetricCompilation<TDirection, TBrand, object, unknown>
