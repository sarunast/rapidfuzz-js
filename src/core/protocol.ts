import type { Direction, MaybeSequence, Sequence } from './types.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

interface Compilation<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly rawScore: (a: Sequence, b: Sequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: (choice: Sequence) => unknown
}

export interface TrustedMetricCompilation<D extends Direction> extends Compilation<D> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

export interface CustomMetricCompilation<D extends Direction> extends Compilation<D> {
  readonly trusted: false
}

export type MetricCompilation<D extends Direction> =
  | TrustedMetricCompilation<D>
  | CustomMetricCompilation<D>
