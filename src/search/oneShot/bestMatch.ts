import { assertOptionKeys } from '#core/options.js'
import { scorerCompilation } from '#core/scoring/scorer.js'
import {
  impossibleThreshold,
  kernelThreshold,
  knownOptimum,
  optionalThreshold,
  passesThreshold,
} from '#core/scoring/threshold.js'
import type { Direction, MaybeSequence } from '#core/types.js'

import type { Match } from '../results.js'
import { assertCollection, collectionEntries } from '../shared/collection.js'
import { BEST_OPTION_KEYS } from '../shared/options.js'
import { choiceReader, normalizeQuery } from '../shared/readers.js'
import type { BestOptions, ItemIterable, Items, AnyMatcherOptions } from '../types.js'
import { arrayItemsOf, better, presentEntries, stableOptionsOf } from './shared.js'

/**
 * The single best match for a query, or `undefined` when nothing clears the
 * threshold — the "did you mean?" shape.
 *
 * ```ts
 * bestMatch('new york jet', teams, { scorer, threshold: 70 })
 * // { item: 'New York Jets', key: 1, score: 72 }
 * ```
 *
 * Always set a threshold. "Best" otherwise means *best available*, so a query
 * with no real match still returns whichever choice is least unlike it — which
 * is how suggestion features end up proposing nonsense.
 *
 * Prepares every choice on every call, which is the right trade for a one-off
 * question and the wrong one for a search box: build a `Matcher` when you can
 * name a second query, or pass `getPrepared` to reuse handles you hold. The
 * scan stops early once a choice hits the scorer's best possible score, since
 * nothing later can beat it.
 *
 * @param query Compared against every choice. Normalized by `normalize` first,
 * if given.
 * @param items Array, `Map`, plain object or any iterable — the shape decides
 * what `key` is on the result.
 * @returns The winning {@link Match}, or `undefined` if the collection is empty
 * or nothing clears the threshold.
 * @throws `TypeError` for an unknown option key, for both `getText` and
 * `getPrepared` at once, for an item the scorer's missing policy refuses, or
 * for a prepared handle from an incompatible scorer.
 * @throws `RangeError` if `threshold` is not a finite number.
 */
export function bestMatch<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: readonly TItem[],
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, number> | undefined
export function bestMatch<TKey, TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ReadonlyMap<TKey, TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, TKey> | undefined
export function bestMatch<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ItemIterable<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, number> | undefined
export function bestMatch<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Readonly<Record<string, TItem>>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, string> | undefined
export function bestMatch<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, unknown> | undefined
export function bestMatch<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): Match<TItem, unknown> | undefined {
  assertOptionKeys(options, BEST_OPTION_KEYS, 'bestMatch')
  const threshold = optionalThreshold(options.threshold)
  assertCollection(items)
  return bestOfCollection(query, items, options, threshold)
}

/**
 * The scan itself, with its options already read.
 *
 * Separate from the public `bestMatch` so `search` can delegate to it at
 * `limit: 1` without its own options being checked a second time — against a
 * key list that does not include `limit`, which is the one key that call is
 * certain to carry.
 */
export function bestOfCollection<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
  threshold: number | null,
): Match<TItem, unknown> | undefined {
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
  const normalized = normalizeQuery(query, normalize)
  const arrayItems = arrayItemsOf(items)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!passesThreshold('similarity', score, threshold)) return undefined
    // One iteration is the point: `presentEntries` is a generator, and this
    // takes its first entry without walking the rest.
    // oxlint-disable-next-line no-unreachable-loop
    for (const { item, key } of presentEntries(items, choices)) {
      return { item, key, score }
    }
    return undefined
  }

  if (impossibleThreshold(compilation, threshold)) return undefined
  const activeThreshold = kernelThreshold(compilation, threshold)

  const prepared = compilation.prepareQuery(normalized)
  const optimal = knownOptimum(compilation)
  let found: Match<TItem, unknown> | undefined
  let cutoff = activeThreshold

  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      if (choice === null) continue
      const score = prepared(choice, cutoff)
      if (!passesThreshold(compilation.direction, score, activeThreshold)) continue
      if (found === undefined || better(compilation.direction, score, found.score)) {
        found = { item, key, score }
        cutoff = score
        if (optimal !== null && score === optimal) break
      }
    }
    return found
  }

  for (const entry of collectionEntries(items)) {
    const choice = choices.read(entry.item)
    if (choice === null) continue
    const score = prepared(choice, cutoff)
    if (!passesThreshold(compilation.direction, score, activeThreshold)) continue
    if (found === undefined || better(compilation.direction, score, found.score)) {
      found = { item: entry.item, key: entry.key, score }
      cutoff = score
      if (optimal !== null && score === optimal) break
    }
  }
  return found
}
