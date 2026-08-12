import type { AnyBrand, PreparedChoice } from '../core/prepared.js'
import type { Scorer } from '../core/scorer.js'
import type { Direction, MaybeSequence, Normalizer } from '../core/types.js'
import type { Match } from './results.js'

export type ItemIterable<TItem> = Iterable<TItem> & object

/**
 * Any collection a search accepts. The shape decides what a result's `key` is:
 * an array gives the index, an iterable the position it was yielded at, a `Map`
 * its own key, a plain object the property name.
 */
export type Items<TItem> =
  | readonly TItem[]
  | ReadonlyMap<unknown, TItem>
  | ItemIterable<TItem>
  | Readonly<Record<string, TItem>>

/**
 * What to do about an item that has no text to score — `null`/`undefined`
 * itself, or whose `getText`/`normalize` returned one.
 *
 * `'skip'` (the default) leaves it out while every other key keeps its place;
 * `'throw'` rejects the collection instead of quietly searching around a hole.
 */
export type MissingItemsPolicy = 'skip' | 'throw'

/** Searching a collection by its text. */
export interface MatcherOptions<
  TItem,
  TDirection extends Direction = Direction,
  TBrand = AnyBrand,
> {
  readonly scorer: Scorer<TDirection, TBrand>
  /**
   * Where the searchable text lives on an item. Required when items are not
   * themselves sequences; results still carry your original item, never the
   * text pulled out of it.
   */
  readonly getText?: ((item: TItem) => MaybeSequence) | undefined
  /**
   * Applied to every choice and to every query, so the two sides can never
   * drift apart. See `normalizeText` for the built-in.
   */
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
  readonly getPrepared?: undefined
}

/**
 * Searching a collection of handles you prepared yourself — mutually exclusive
 * with {@link MatcherOptions}, because a prepared row has no text to extract
 * and no gap to skip.
 */
export interface PreparedMatcherOptions<
  TItem,
  TDirection extends Direction = Direction,
  TBrand = AnyBrand,
> {
  readonly scorer: Scorer<TDirection, TBrand>
  /**
   * Pulls the handle off a row. Each must have been made by a compatible
   * scorer, which is checked at runtime and, for a built-in metric, at compile
   * time through the handle's brand.
   */
  readonly getPrepared: (item: TItem) => PreparedChoice<NoInfer<TBrand>>
  /**
   * Applied to the **query only** — the choices were normalized when they were
   * prepared. It has to be the same function passed to `prepareChoice`, by
   * identity, or the search refuses the handle.
   */
  readonly normalize?: Normalizer | undefined
  readonly getText?: undefined
  readonly missingItems?: undefined
}

export type AnyMatcherOptions<
  TItem,
  TDirection extends Direction = Direction,
  TBrand = AnyBrand,
> =
  | MatcherOptions<TItem, TDirection, TBrand>
  | PreparedMatcherOptions<TItem, TDirection, TBrand>

/**
 * Searching a collection through one corpus-wide index — {@link MatcherOptions}
 * with the two things an index cannot do taken away.
 *
 * The scorer must be a **similarity** scorer, which is a compile error rather
 * than a throw: an index accumulates how much two sequences share, and ranking
 * by a distance is a different search that nothing has measured. Whether that
 * scorer has an indexed representation at all is checked when the matcher is
 * built, because a custom scorer cannot be recognised by its type.
 *
 * `getPrepared` is refused for the same reason it exists: a prepared handle is
 * the per-choice representation an index replaces.
 */
export interface IndexedMatcherOptions<TItem, TBrand = AnyBrand> {
  /** A similarity scorer that offers an index — `dice` or `cosine` today. */
  readonly scorer: Scorer<'similarity', TBrand>
  /**
   * Where the searchable text lives on an item. Required when items are not
   * themselves sequences; results still carry your original item.
   */
  readonly getText?: ((item: TItem) => MaybeSequence) | undefined
  /**
   * Applied to every choice and to every query, so the two sides can never
   * drift apart. See `normalizeText` for the built-in.
   */
  readonly normalize?: Normalizer | undefined
  /** What to do with an item that has no text; `'skip'` by default. */
  readonly missingItems?: MissingItemsPolicy | undefined
  /** Not supported: an index replaces the handles this would supply. */
  readonly getPrepared?: undefined
}

/**
 * What a reader is built from: both accessors at once, which the public union
 * refuses and a JavaScript caller can still pass. Reading them by value and
 * refusing the combination is `choiceReader`'s job, so the shape it takes has
 * to be able to hold one.
 */
export interface ResolvedMatcherOptions<
  TItem,
  TDirection extends Direction = Direction,
  TBrand = AnyBrand,
> {
  readonly scorer: Scorer<TDirection, TBrand>
  readonly getText?: ((item: TItem) => MaybeSequence) | undefined
  readonly getPrepared?: ((item: TItem) => PreparedChoice<NoInfer<TBrand>>) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

/** Accepted by every single-result and streaming search. */
export interface BestOptions {
  /**
   * The quality bar, on the scorer's own scale. Without one, the least-bad
   * candidate still wins — which is how "did you mean?" features end up
   * suggesting nonsense.
   */
  readonly threshold?: number | undefined
}

/** {@link BestOptions} plus the cap on how many results come back. */
export interface SearchOptions extends BestOptions {
  /** Defaults to five; `null` returns every qualifying match. */
  readonly limit?: number | null | undefined
}

/**
 * A collection with every choice prepared once, at construction, so each later
 * query pays only for itself. Build one whenever you can name a second query
 * over the same data.
 *
 * A Matcher snapshots what it *scores*, not what it returns: pushing to the
 * source collection afterwards does not change results, while returned items
 * stay live references to your own objects.
 */
export interface Matcher<
  TItem,
  TKey,
  TDirection extends Direction = Direction,
  TBrand = AnyBrand,
> {
  /** How many choices are searchable — gaps that were skipped are not counted. */
  readonly size: number
  // Branded like the scorer it was built from, so a handle made through
  // `matcher.scorer` keeps the compile-time half of its identity.
  readonly scorer: Scorer<TDirection, TBrand>
  /** The single best match, or `undefined` when nothing clears the threshold. */
  readonly best: (
    query: MaybeSequence,
    options?: BestOptions,
  ) => Match<TItem, TKey> | undefined
  /** Up to `limit` matches, best first. */
  readonly search: (
    query: MaybeSequence,
    options?: SearchOptions,
  ) => readonly Match<TItem, TKey>[]
  /**
   * Every qualifying match, streamed in collection order rather than by score —
   * which is why it takes no `limit`.
   *
   * How much is done before the first result depends on how the matcher was
   * built: `createMatcher` scores lazily, so stopping early leaves the rest
   * unscored, while an indexed matcher settles the qualifying set up front. The
   * values and their order are the same either way.
   */
  readonly searchIter: (
    query: MaybeSequence,
    options?: BestOptions,
  ) => IterableIterator<Match<TItem, TKey>>
}
