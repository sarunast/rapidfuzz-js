import type { Scorer } from '../core/scorer.js'
import type { Direction, MaybeSequence, Sequence } from '../core/types.js'
import type { Match } from './results.js'

export type Items<T> =
  | readonly T[]
  | ReadonlyMap<unknown, T>
  | Iterable<T>
  | Readonly<Record<string, T>>

export type Normalizer = (value: Sequence) => MaybeSequence
export type MissingItemsPolicy = 'skip' | 'throw'

export interface MatcherOptions<T, D extends Direction> {
  readonly scorer: Scorer<D>
  readonly getText?: ((item: T) => MaybeSequence) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

export interface BestOptions {
  readonly threshold?: number | undefined
}

export interface SearchOptions extends BestOptions {
  readonly limit?: number | null | undefined
}

export interface Matcher<T, K, D extends Direction = Direction> {
  readonly size: number
  readonly scorer: Scorer<D>
  best(query: MaybeSequence, options?: BestOptions): Match<T, K> | undefined
  search(query: MaybeSequence, options?: SearchOptions): readonly Match<T, K>[]
}
