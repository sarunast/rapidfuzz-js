import type { Direction, MaybeSequence, Sequence } from './types.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

/**
 * A scorer with one side already held, ready to be run against many choices.
 *
 * `choice` is whatever {@link Compilation.prepareChoice} returned — opaque here
 * on purpose. Only the algorithm that produced it knows what it is, and only it
 * unwraps and checks it, which is what keeps a second type parameter for the
 * prepared form out of `Scorer`, the batch entry points and the matcher.
 *
 * `threshold` is `null` for "no active cutoff", which is not the same as `0`:
 * zero is a real cutoff a score can fail, and `null` says the kernel may skip
 * whatever pruning a cutoff would have paid for. `trustedKernelThreshold` turns
 * a threshold that cannot reject anything into exactly this `null`.
 */
export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

interface Compilation<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  /**
   * Score at the public boundary: validates the pair and applies the missing
   * policy before any kernel sees it.
   */
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  /**
   * Score a pair whose sequence contract a caller has already established, so
   * that a batch of thousands pays that check once rather than per cell.
   */
  readonly rawScore: (a: Sequence, b: Sequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  /** The opaque value a {@link PreparedKernel} takes. See its note. */
  readonly prepareChoice: (choice: Sequence) => unknown
}

/**
 * A compilation this package built, whose bounds and cutoff behaviour the
 * orchestration layers may reason about — that is what `trusted` licenses.
 *
 * `validate` is the half of {@link Compilation.score} that is left when the
 * score itself is already known: a threshold no score can fail still owes the
 * caller the error an illegal input would have raised.
 */
export interface TrustedMetricCompilation<D extends Direction> extends Compilation<D> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

/**
 * A compilation around a caller's own metric. It carries no `validate`, because
 * nothing here may assume what that function does with its bounds — every score
 * has to be observed rather than predicted.
 */
export interface CustomMetricCompilation<D extends Direction> extends Compilation<D> {
  readonly trusted: false
}

export type MetricCompilation<D extends Direction> =
  | TrustedMetricCompilation<D>
  | CustomMetricCompilation<D>
