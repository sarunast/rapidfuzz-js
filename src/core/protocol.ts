import type { Direction, MaybeSequence, Sequence } from './types.js'

export const METRIC_CONFIGURATION: unique symbol = Symbol(
  'rapidfuzz.metric.configuration',
)
export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

export interface MetricCompilation<D extends Direction> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly trusted: boolean
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: (choice: Sequence) => unknown
}
