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
  // Borrows the sequence: the result lives no longer than the loop that made
  // it, so preparation may keep a reference into the caller's data.
  readonly prepareChoice: (choice: Sequence) => unknown
  // Owns the sequence, for a handle that outlives the call. Whether that costs
  // a copy is the preparation's to know — most already copy what they convert.
  readonly prepareOwnedChoice: (choice: Sequence) => unknown
  // Identity a prepared choice is checked against; shared by compilations
  // whose preparation is compatible, fresh when the recorded configuration
  // differs or for a custom scorer. `missing` keeps the shared one — it
  // decides which pairs are refused and never reaches preparation.
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
