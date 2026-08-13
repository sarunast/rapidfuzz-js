import { assertOptionKeys } from '../core/options.js'
import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence } from '../core/types.js'
import type { ChoiceTable } from './choiceTable.js'
import { buildChoiceTable, matchAt } from './choiceTable.js'
import { assertCollection } from './collection.js'
import { bestDistance } from './internal/bestDistance.js'
import { bestSimilarity } from './internal/bestSimilarity.js'
import { topDistance } from './internal/topDistance.js'
import { topSimilarity } from './internal/topSimilarity.js'
import type { ScoredId } from './internal/types.js'
import type { Match } from './results.js'
import {
  CALL_BEST_KEYS,
  CALL_SEARCH_KEYS,
  MATCHER_OPTION_KEYS,
  choiceReader,
  normalizeQuery,
  optionalThreshold,
  resultLimit,
} from './snapshot.js'
import type {
  AnyMatcherOptions,
  ResolvedMatcherOptions,
  BestOptions,
  ItemIterable,
  Items,
  Matcher,
  SearchOptions,
} from './types.js'

// Both helpers answer a missing query, which only a similarity scorer accepts:
// a distance metric refuses the pair in `validatePair` before a score exists.
// That is why they qualify with `score < threshold` rather than reading the
// direction — under distance the call has already thrown.
function missingSimilarityBest<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
): Match<TItem, unknown> | undefined {
  if (threshold !== null && score < threshold) return undefined
  return table.items.length === 0 ? undefined : matchAt(table, 0, score)
}

function missingSimilarityTop<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
  limit: number | null,
): readonly Match<TItem, unknown>[] {
  if (threshold !== null && score < threshold) return []
  const count = table.items.length
  const length = limit === null ? count : Math.min(count, limit)
  const matches: Match<TItem, unknown>[] = new Array(length)
  for (let id = 0; id < length; id++) matches[id] = matchAt(table, id, score)
  return matches
}

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
  const optimal = compilation.trusted
    ? direction === 'similarity'
      ? compilation.bounds[1]
      : compilation.bounds[0]
    : null
  // A copy, so a caller who mutates their options object afterwards cannot
  // change a matcher that has already read them. The properties are declared
  // as `| undefined`, so naming an absent one costs nothing.
  // Both accessors travel through, so a JavaScript caller who passed each kind
  // is refused by the reader rather than quietly served the prepared one.
  const stableOptions: ResolvedMatcherOptions<TItem, Direction, TBrand> = {
    scorer,
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
  const { table, values: prepared } = buildChoiceTable(items, choices.read)

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
    if (
      compilation.trusted &&
      impossibleTrustedThreshold(direction, compilation.bounds, threshold)
    ) {
      return undefined
    }
    const activeThreshold = compilation.trusted
      ? trustedKernelThreshold(direction, compilation.bounds, threshold)
      : threshold
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
    if (
      compilation.trusted &&
      impossibleTrustedThreshold(direction, compilation.bounds, threshold)
    ) {
      return []
    }
    const activeThreshold = compilation.trusted
      ? trustedKernelThreshold(direction, compilation.bounds, threshold)
      : threshold
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
        if (threshold !== null && missingScore < threshold) return
        for (let id = 0; id < table.items.length; id++) {
          yield matchAt(table, id, missingScore)
        }
        return
      }
      if (
        compilation.trusted &&
        impossibleTrustedThreshold(direction, compilation.bounds, threshold)
      ) {
        return
      }
      const activeThreshold = compilation.trusted
        ? trustedKernelThreshold(direction, compilation.bounds, threshold)
        : threshold
      const score = compilation.prepareQuery(normalized)
      const similarity = direction === 'similarity'
      for (let id = 0; id < prepared.length; id++) {
        const value = score(prepared[id], activeThreshold)
        if (
          threshold === null ||
          (similarity ? value >= threshold : value <= threshold)
        ) {
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
