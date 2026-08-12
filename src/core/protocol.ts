import type { AnyBrand } from './prepared.js'
import type { Direction, MaybeSequence, Sequence } from './types.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

interface Compilation<D extends Direction, Brand = AnyBrand> {
  readonly direction: D
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly rawScore: (a: Sequence, b: Sequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  readonly prepareChoice: (choice: Sequence) => unknown
  // Identity a prepared choice is checked against; shared by every default
  // compilation of one metric, fresh for a configured or custom one.
  readonly preparedChoiceKey: object
  // Never assigned and never read: it exists so a scorer's handles carry the
  // metric that made them into the type system.
  readonly preparedChoiceBrand?: Brand
}

export interface TrustedMetricCompilation<
  D extends Direction,
  Brand = AnyBrand,
> extends Compilation<D, Brand> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

export interface CustomMetricCompilation<
  D extends Direction,
  Brand = AnyBrand,
> extends Compilation<D, Brand> {
  readonly trusted: false
}

export type MetricCompilation<D extends Direction, Brand = AnyBrand> =
  | TrustedMetricCompilation<D, Brand>
  | CustomMetricCompilation<D, Brand>
