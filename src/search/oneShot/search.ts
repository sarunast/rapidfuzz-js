import { assertOptionKeys } from '../../core/options.js'
import { scorerCompilation } from '../../core/scorer.js'
import {
  impossibleThreshold,
  kernelThreshold,
  knownOptimum,
  optionalThreshold,
  passesThreshold,
} from '../../core/threshold.js'
import type { Direction, MaybeSequence } from '../../core/types.js'
import type { Match } from '../results.js'
import { assertCollection, collectionEntries } from '../shared/collection.js'
import { pushHeap, replaceHeapRoot } from '../shared/heap.js'
import { SEARCH_OPTION_KEYS, resultLimit } from '../shared/options.js'
import { choiceReader, normalizeQuery } from '../shared/readers.js'
import type { ItemIterable, Items, AnyMatcherOptions, SearchOptions } from '../types.js'
import { bestOfCollection } from './bestMatch.js'
import { arrayItemsOf, better, presentEntries, stableOptionsOf } from './shared.js'

/**
 * A scored candidate with the source position that breaks its ties.
 *
 * Carried here rather than as a choice id: a one-shot search skips over gaps as
 * it reads, so the position a result reports is not the count of what it kept.
 */
interface ScoredEntry<TItem, TKey> extends Match<TItem, TKey> {
  readonly order: number
}

function worse<TItem, TKey>(
  direction: Direction,
  left: ScoredEntry<TItem, TKey>,
  right: ScoredEntry<TItem, TKey>,
): boolean {
  if (left.score !== right.score) return better(direction, right.score, left.score)
  return left.order > right.order
}

function orderedResults<TItem, TKey>(
  direction: Direction,
  entries: ScoredEntry<TItem, TKey>[],
): readonly Match<TItem, TKey>[] {
  entries.sort((a, b) => {
    const byScore = direction === 'similarity' ? b.score - a.score : a.score - b.score
    return byScore || a.order - b.order
  })
  return entries.map(({ item, key, score }) => ({ item, key, score }))
}

/**
 * A ranked list of matches, best first — autocomplete dropdowns and results
 * pages.
 *
 * ```ts
 * search('new york', teams, { scorer, threshold: 60, limit: 2 })
 * // [ { item: 'New York Jets', key: 1, score: 68.4 },
 * //   { item: 'New York Giants', key: 2, score: 68.4 } ]
 * ```
 *
 * The two options compose: `threshold` decides what deserves to appear at all,
 * `limit` how many you show. Ties keep the order the collection had.
 *
 * Use `searchIter` when you want to stop early instead of ranking everything,
 * and `bestMatch` when one winner is enough.
 *
 * @param query Compared against every choice.
 * @param items Array, `Map`, plain object or any iterable — the shape decides
 * what `key` is on each result.
 * @returns Up to `limit` {@link Match} results, best first. Empty when nothing
 * clears the threshold. `limit` defaults to `5`; `null` returns every
 * qualifying match.
 * @throws `TypeError` for an unknown option key, for both `getText` and
 * `getPrepared` at once, for an item the scorer's missing policy refuses, or
 * for a prepared handle from an incompatible scorer.
 * @throws `RangeError` if `threshold` is not finite, or `limit` is negative or
 * not an integer.
 */
export function search<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: readonly TItem[],
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & SearchOptions,
): readonly Match<TItem, number>[]
export function search<TKey, TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ReadonlyMap<TKey, TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & SearchOptions,
): readonly Match<TItem, TKey>[]
export function search<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ItemIterable<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & SearchOptions,
): readonly Match<TItem, number>[]
export function search<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Readonly<Record<string, TItem>>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & SearchOptions,
): readonly Match<TItem, string>[]
export function search<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & SearchOptions,
): readonly Match<TItem, unknown>[] {
  assertOptionKeys(options, SEARCH_OPTION_KEYS, 'search')
  // Argument shape is checked before any semantic exit: `limit: 0` must not
  // excuse an invalid collection or a non-finite threshold.
  const limit = resultLimit(options.limit)
  const threshold = optionalThreshold(options.threshold)
  assertCollection(items)
  if (limit === 1) {
    const match = bestOfCollection(query, items, options, threshold)
    return match === undefined ? [] : [match]
  }
  const scorer = options.scorer
  const compilation = scorerCompilation(scorer)
  const normalize = options.normalize
  const stableOptions = stableOptionsOf(options, normalize)
  const choices = choiceReader(
    stableOptions,
    compilation.prepareChoice,
    compilation.preparedChoiceKey,
    false,
  )
  // Every option is read before the exit, so `limit: 0` refuses a foreign
  // scorer and an unknown `missingItems` the way every other limit does. What
  // it still skips is the work: no query normalization, no traversal.
  if (limit === 0) return []
  const normalized = normalizeQuery(query, normalize)
  const arrayItems = arrayItemsOf(items)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!passesThreshold('similarity', score, threshold)) return []
    const results: Match<TItem, unknown>[] = []
    for (const { item, key } of presentEntries(items, choices)) {
      results.push({ item, key, score })
      if (limit !== null && results.length === limit) break
    }
    return results
  }

  if (impossibleThreshold(compilation, threshold)) return []
  const activeThreshold = kernelThreshold(compilation, threshold)

  const prepared = compilation.prepareQuery(normalized)
  const results: ScoredEntry<TItem, unknown>[] = []
  const heapWorse = (
    left: ScoredEntry<TItem, unknown>,
    right: ScoredEntry<TItem, unknown>,
  ) => worse(compilation.direction, left, right)
  // Once a full heap holds nothing but optimal scores, later candidates can
  // only tie, and a tie loses on order — so the scan is finished. The Matcher
  // drivers stop on the same condition.
  const optimal = knownOptimum(compilation)
  let cutoff = activeThreshold

  if (arrayItems !== null) {
    // An array index is already the source order the heap breaks ties on, so
    // the counter the generic branch keeps is one the array branch can read.
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      if (choice === null) continue
      const score = prepared(choice, cutoff)
      if (passesThreshold(compilation.direction, score, activeThreshold)) {
        if (limit === null) {
          results.push({ item, key, score, order: key })
        } else if (results.length < limit) {
          pushHeap(results, { item, key, score, order: key }, heapWorse)
          if (results.length === limit) {
            cutoff = results[0].score
            if (optimal !== null && cutoff === optimal) break
          }
        } else if (better(compilation.direction, score, results[0].score)) {
          replaceHeapRoot(results, { item, key, score, order: key }, heapWorse)
          cutoff = results[0].score
          if (optimal !== null && cutoff === optimal) break
        }
      }
    }
  } else {
    let order = 0
    for (const entry of collectionEntries(items)) {
      const choice = choices.read(entry.item)
      if (choice === null) continue
      const score = prepared(choice, cutoff)
      if (passesThreshold(compilation.direction, score, activeThreshold)) {
        if (limit === null) {
          results.push({ item: entry.item, key: entry.key, score, order })
        } else if (results.length < limit) {
          pushHeap(results, { item: entry.item, key: entry.key, score, order }, heapWorse)
          if (results.length === limit) {
            cutoff = results[0].score
            if (optimal !== null && cutoff === optimal) break
          }
        } else if (better(compilation.direction, score, results[0].score)) {
          replaceHeapRoot(
            results,
            { item: entry.item, key: entry.key, score, order },
            heapWorse,
          )
          cutoff = results[0].score
          if (optimal !== null && cutoff === optimal) break
        }
      }
      order++
    }
  }
  return orderedResults(compilation.direction, results)
}
