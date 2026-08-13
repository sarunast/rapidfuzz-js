import { assertOptionKeys } from '../../core/options.js'
import type { OptimumProof } from '../../core/scoring/optimumProof.js'
import { scorerCompilation } from '../../core/scoring/scorer.js'
import {
  impossibleThreshold,
  kernelThreshold,
  knownOptimum,
  optionalThreshold,
  passesThreshold,
} from '../../core/scoring/threshold.js'
import type { Direction, MaybeSequence } from '../../core/types.js'
import type { Match } from '../results.js'
import { assertCollection } from '../shared/collection.js'
import {
  CALL_BEST_KEYS,
  CALL_SEARCH_KEYS,
  MATCHER_OPTION_KEYS,
  resultLimit,
} from '../shared/options.js'
import { choiceReader, normalizeQuery } from '../shared/readers.js'
import type { ReaderOptions } from '../shared/readers.js'
import type {
  AnyMatcherOptions,
  BestOptions,
  ItemIterable,
  Items,
  Matcher,
  SearchOptions,
} from '../types.js'
import { buildChoiceTable, matchAt } from './choiceTable.js'
import {
  missingSimilarityBest,
  missingSimilarityMatches,
  missingSimilarityTop,
} from './missingQuery.js'
import { bestDistance } from './scan/bestDistance.js'
import { bestSimilarity } from './scan/bestSimilarity.js'
import { topDistance } from './scan/topDistance.js'
import { topSimilarity } from './scan/topSimilarity.js'
import type { ScoredId } from './scan/types.js'

/**
 * Prepare a collection once and query it many times.
 *
 * Construction is where every choice is converted into the form the scorer's
 * kernels want; each later `best`/`search`/`searchIter` call then pays only for
 * its own query. On a 2,000-title token-sort workload that measured 6.63×
 * faster than the one-shot search it replaces.
 *
 * ```ts
 * const matcher = createMatcher(products, {
 *   scorer,
 *   getText: (product) => product.title,
 *   normalize: normalizeText,
 * })
 * matcher.best('mechanical keybord', { threshold: 70 })
 * ```
 *
 * What it snapshots is what it *scores*, not what it returns: strings are kept
 * as-is and other sequences shallow-copied, so pushing to the source array
 * afterwards does not change results — while returned items stay live
 * references to your own objects. Rebuild when the collection changes;
 * construction is the cheap part next to re-preparing on every query.
 *
 * Items with no text — `null`/`undefined`, or whose `getText` or `normalize`
 * returns one — are skipped while every other key keeps its place, so item 3
 * stays item 3 even if item 2 was a gap.
 *
 * Pass `getPrepared` instead of `getText` to supply handles you prepared
 * yourself, which are then resolved once here rather than prepared at all.
 *
 * `searchIter` here is lazy in the strong sense: it scores in collection order
 * as the iterator advances, so a caller who stops early pays only for what they
 * consumed. That is a property of this constructor rather than of the
 * {@link Matcher} interface — `createIndexedMatcher` returns the same values in
 * the same order, having settled them before the first one.
 *
 * @param items Array, `Map`, plain object or any iterable — the shape decides
 * what `key` is on every result this Matcher returns.
 * @returns A frozen {@link Matcher} exposing `best`, `search`, `searchIter`,
 * `size` and the `scorer` it was built from.
 * @throws `TypeError` for an unknown option key, for both `getText` and
 * `getPrepared` at once, for a gap when `missingItems: 'throw'`, or for a
 * prepared handle from an incompatible scorer.
 */
export function createMatcher<TItem, TDirection extends Direction, TBrand>(
  items: readonly TItem[],
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
): Matcher<TItem, number, TDirection, TBrand>
export function createMatcher<TKey, TItem, TDirection extends Direction, TBrand>(
  items: ReadonlyMap<TKey, TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
): Matcher<TItem, TKey, TDirection, TBrand>
export function createMatcher<TItem, TDirection extends Direction, TBrand>(
  items: ItemIterable<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
): Matcher<TItem, number, TDirection, TBrand>
export function createMatcher<TItem, TDirection extends Direction, TBrand>(
  items: Readonly<Record<string, TItem>>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
): Matcher<TItem, string, TDirection, TBrand>
export function createMatcher<TItem, TDirection extends Direction, TBrand>(
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
): Matcher<TItem, unknown, TDirection, TBrand> {
  assertOptionKeys(options, MATCHER_OPTION_KEYS, 'createMatcher')
  // Read exactly once each: a getter or proxy could otherwise answer one thing
  // to the reader and another to the copy below, leaving a matcher whose
  // choices and queries were normalized by different functions.
  const scorer = options.scorer
  const normalize = options.normalize
  const getPrepared = options.getPrepared
  const getText = options.getText
  const missingItems = options.missingItems
  // Same order as the one-shot entry points: the collection is checked before
  // anything semantic, so a wrong argument is refused the same way whichever
  // API the caller reached for.
  assertCollection(items)
  const compilation = scorerCompilation(scorer)
  // Fixed for the matcher's lifetime: direction and bounds belong to the
  // scorer, and only the threshold changes from one call to the next.
  const direction = compilation.direction
  const optimal = knownOptimum(compilation)
  // A copy, so a caller who mutates their options object afterwards cannot
  // change a matcher that has already read them. The properties are declared
  // as `| undefined`, so naming an absent one costs nothing.
  // Both accessors travel through, so a JavaScript caller who passed each kind
  // is refused by the reader rather than quietly served the prepared one.
  const stableOptions: ReaderOptions<TItem, TBrand> = {
    getText,
    getPrepared,
    normalize,
    missingItems,
  }
  // Every handle is resolved here, once, so a query pays nothing for the mode
  // it was built in — the drivers read `prepared` the same way either way.
  const choices = choiceReader(
    stableOptions,
    compilation.prepareChoice,
    compilation.preparedChoiceKey,
    true,
  )
  // `prepared[id]` and `table.items[id]` are the same choice. The corpus-wide
  // index `createIndexedMatcher` builds sits in exactly this position, which is
  // the whole point of the two sharing a table.
  //
  // Sized from the source where the source has a size, because growing it one
  // choice at a time is what a flat layout otherwise pays for the second array.
  const prepared: unknown[] = Array.isArray(items) ? new Array(items.length) : []
  const table = buildChoiceTable(items, (item, id) => {
    const value = choices.read(item)
    if (value === null) return false
    prepared[id] = value
    return true
  })
  prepared.length = table.items.length

  // Built on the first query that could actually use it, never at construction:
  // a metric offering a proof is not a caller wanting one, and a collection that
  // is only ever streamed through `searchIter` must not pay to index itself.
  // `null` records that there is nothing to build, so the test happens once.
  let proof: OptimumProof | null | undefined
  const optimumProof = (): OptimumProof | null => {
    if (proof === undefined) {
      const factory = compilation.proveOptimum
      proof =
        factory === undefined || optimal === null || direction !== 'similarity'
          ? null
          : factory(prepared)
    }
    return proof
  }
  // A threshold above the optimum admits nothing, and one at or below it admits
  // every optimal choice — so the proof applies exactly when the threshold does
  // not exclude the optimum. `impossibleThreshold` has already refused the
  // first case by the time either caller reaches this.
  const proofApplies = (threshold: number | null): boolean =>
    threshold === null || optimal === null || threshold <= optimal

  const materialize = (found: readonly ScoredId[]): readonly Match<TItem, unknown>[] => {
    const matches: Match<TItem, unknown>[] = new Array(found.length)
    for (let at = 0; at < found.length; at++) {
      matches[at] = matchAt(table, found[at].id, found[at].score)
    }
    return matches
  }

  const best = (
    query: MaybeSequence,
    call?: BestOptions,
  ): Match<TItem, unknown> | undefined => {
    // Guarded rather than checked unconditionally: an absent options object is
    // the common call, and it has no keys to walk.
    if (call !== undefined) assertOptionKeys(call, CALL_BEST_KEYS, 'matcher.best')
    const threshold = optionalThreshold(call?.threshold)
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      const missingScore = compilation.score(query, '', threshold)
      return missingSimilarityBest(table, missingScore, threshold)
    }
    if (impossibleThreshold(compilation, threshold)) return undefined
    if (proofApplies(threshold) && optimal !== null) {
      // Ahead of `prepareQuery`, so a settled query never builds the kernel
      // state it would have scored with.
      const settled = optimumProof()?.best(normalized)
      if (settled !== undefined) return matchAt(table, settled, optimal)
    }
    const activeThreshold = kernelThreshold(compilation, threshold)
    const score = compilation.prepareQuery(normalized)
    const found =
      direction === 'similarity'
        ? bestSimilarity(prepared, score, activeThreshold, optimal)
        : bestDistance(prepared, score, activeThreshold, optimal)
    return found === undefined ? undefined : matchAt(table, found.id, found.score)
  }
  const search = (
    query: MaybeSequence,
    call?: SearchOptions,
  ): readonly Match<TItem, unknown>[] => {
    if (call !== undefined) assertOptionKeys(call, CALL_SEARCH_KEYS, 'matcher.search')
    const limit = resultLimit(call?.limit)
    const threshold = optionalThreshold(call?.threshold)
    if (limit === 0) return []
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      const missingScore = compilation.score(query, '', threshold)
      return missingSimilarityTop(table, missingScore, threshold, limit)
    }
    if (impossibleThreshold(compilation, threshold)) return []
    // Only a limited search: an unlimited one has to report the choices below
    // the optimum too, which the proof knows nothing about. `resultLimit` gives
    // an absent `limit` the default 5, so a bare `search()` does reach here —
    // `limit: null` is the one spelling that never does.
    if (limit !== null && proofApplies(threshold) && optimal !== null) {
      const settled = optimumProof()?.top(normalized, limit)
      if (settled !== undefined) {
        const matches: Match<TItem, unknown>[] = new Array(settled.length)
        for (let at = 0; at < settled.length; at++) {
          matches[at] = matchAt(table, settled[at], optimal)
        }
        return matches
      }
    }
    const activeThreshold = kernelThreshold(compilation, threshold)
    const score = compilation.prepareQuery(normalized)
    return materialize(
      direction === 'similarity'
        ? topSimilarity(prepared, score, activeThreshold, limit, optimal)
        : topDistance(prepared, score, activeThreshold, limit, optimal),
    )
  }
  const searchIter = (
    query: MaybeSequence,
    call?: BestOptions,
  ): IterableIterator<Match<TItem, unknown>> => {
    // Read where the call is made, not where iteration starts: a caller who
    // mutates their options object between the two would otherwise change a
    // search already asked for. Scoring stays lazy; only the number is taken.
    if (call !== undefined) assertOptionKeys(call, CALL_BEST_KEYS, 'matcher.searchIter')
    const threshold = optionalThreshold(call?.threshold)
    function* iterate(): Generator<Match<TItem, unknown>> {
      const normalized = normalizeQuery(query, normalize)
      if (normalized === null) {
        const missingScore = compilation.score(query, '', threshold)
        yield* missingSimilarityMatches(table, missingScore, threshold)
        return
      }
      if (impossibleThreshold(compilation, threshold)) return
      const activeThreshold = kernelThreshold(compilation, threshold)
      const score = compilation.prepareQuery(normalized)
      for (let id = 0; id < prepared.length; id++) {
        const value = score(prepared[id], activeThreshold)
        if (passesThreshold(direction, value, threshold)) {
          yield matchAt(table, id, value)
        }
      }
    }

    return iterate()
  }
  return Object.freeze({
    size: table.items.length,
    scorer,
    best,
    search,
    searchIter,
  })
}
