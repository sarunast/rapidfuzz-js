import type { AnyBrand, PreparedChoice } from '../core/prepared.js'
import type { Scorer } from '../core/scorer.js'
import type { Direction, MaybeSequence, Normalizer } from '../core/types.js'
import type { Match } from './results.js'

export type ItemIterable<T> = Iterable<T> & object

export type Items<T> =
  | readonly T[]
  | ReadonlyMap<unknown, T>
  | ItemIterable<T>
  | Readonly<Record<string, T>>

export type MissingItemsPolicy = 'skip' | 'throw'

export interface MatcherOptions<T, D extends Direction = Direction> {
  readonly scorer: Scorer<D>
  readonly getText?: ((item: T) => MaybeSequence) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
  readonly getPrepared?: undefined
}

export interface PreparedMatcherOptions<
  T,
  D extends Direction = Direction,
  B = AnyBrand,
> {
  readonly scorer: Scorer<D, B>
  readonly getPrepared: (item: T) => PreparedChoice<NoInfer<B>>
  readonly normalize?: Normalizer | undefined
  readonly getText?: undefined
  readonly missingItems?: undefined
}

export type AnyMatcherOptions<T, D extends Direction = Direction, B = AnyBrand> =
  | MatcherOptions<T, D>
  | PreparedMatcherOptions<T, D, B>

export interface BestOptions {
  readonly threshold?: number | undefined
}

export interface SearchOptions extends BestOptions {
  /** Defaults to five; `null` returns every qualifying match. */
  readonly limit?: number | null | undefined
}

export interface Matcher<T, K, D extends Direction = Direction> {
  readonly size: number
  readonly scorer: Scorer<D>
  readonly best: (query: MaybeSequence, options?: BestOptions) => Match<T, K> | undefined
  readonly search: (
    query: MaybeSequence,
    options?: SearchOptions,
  ) => readonly Match<T, K>[]
  readonly searchIter: (
    query: MaybeSequence,
    options?: BestOptions,
  ) => IterableIterator<Match<T, K>>
}
