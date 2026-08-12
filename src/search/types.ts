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

export interface MatcherOptions<T, D extends Direction = Direction, B = AnyBrand> {
  readonly scorer: Scorer<D, B>
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
  | MatcherOptions<T, D, B>
  | PreparedMatcherOptions<T, D, B>

/**
 * What a reader is built from: both accessors at once, which the public union
 * refuses and a JavaScript caller can still pass. Reading them by value and
 * refusing the combination is `choiceReader`'s job, so the shape it takes has
 * to be able to hold one.
 */
export interface ResolvedMatcherOptions<
  T,
  D extends Direction = Direction,
  B = AnyBrand,
> {
  readonly scorer: Scorer<D, B>
  readonly getText?: ((item: T) => MaybeSequence) | undefined
  readonly getPrepared?: ((item: T) => PreparedChoice<NoInfer<B>>) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

export interface BestOptions {
  readonly threshold?: number | undefined
}

export interface SearchOptions extends BestOptions {
  /** Defaults to five; `null` returns every qualifying match. */
  readonly limit?: number | null | undefined
}

export interface Matcher<T, K, D extends Direction = Direction, B = AnyBrand> {
  readonly size: number
  // Branded like the scorer it was built from, so a handle made through
  // `matcher.scorer` keeps the compile-time half of its identity.
  readonly scorer: Scorer<D, B>
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
