import { assertOptionKeys } from '#core/options.js'
import type { AnyMetricCompilation, PreparedKernel } from '#core/scoring/compilation.js'
import { scorerCompilation } from '#core/scoring/scorer.js'
import {
  impossibleThreshold,
  kernelThreshold,
  optionalThreshold,
  passesThreshold,
} from '#core/scoring/threshold.js'
import type { Direction, MaybeSequence, Normalizer } from '#core/types.js'

import type { Match } from '../results.js'
import { assertCollection, collectionEntries } from '../shared/collection.js'
import { BEST_OPTION_KEYS } from '../shared/options.js'
import { choiceReader, normalizeQuery } from '../shared/readers.js'
import type { ChoiceReader } from '../shared/readers.js'
import type { BestOptions, ItemIterable, Items, AnyMatcherOptions } from '../types.js'
import { arrayItemsOf, presentEntries, stableOptionsOf } from './shared.js'

const STREAM_PREPARE_AFTER = 8

/**
 * Every qualifying match, streamed lazily in **collection order** rather than
 * by score — which is why it takes no `limit`.
 *
 * Candidates are pulled one at a time, so stopping early leaves the rest
 * unscored. Pair it with a generator to run cheap guards before any scoring:
 *
 * ```ts
 * for (const match of searchIter(query, plausible(query), {
 *   scorer,
 *   getPrepared: (row) => row.prepared,
 *   threshold: 80,
 * })) {
 *   return match // the rest of the collection is never guarded or scored
 * }
 * ```
 *
 * Two things to know when the source is a generator: `key` counts what was
 * *yielded*, so filtering renumbers it and you should carry your own id on the
 * item; and a generator is consumed once, so call the generator function per
 * query rather than reusing its result.
 *
 * Use `search` when you need results ranked, and `bestMatch` when you want the
 * highest-scoring one rather than the first acceptable one.
 *
 * @param query Compared against every choice.
 * @param items Array, `Map`, plain object or any iterable, pulled on demand.
 * @returns An iterator of {@link Match} in collection order, yielding only what
 * clears the threshold.
 * @throws `TypeError` for an unknown option key — note `limit` is not one of
 * them — for both `getText` and `getPrepared` at once, for an item the scorer's
 * missing policy refuses, or for a prepared handle from an incompatible scorer.
 * @throws `RangeError` if `threshold` is not a finite number.
 */
export function searchIter<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: readonly TItem[],
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, number>>
export function searchIter<TKey, TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ReadonlyMap<TKey, TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, TKey>>
export function searchIter<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: ItemIterable<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, number>>
export function searchIter<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Readonly<Record<string, TItem>>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, string>>
export function searchIter<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, unknown>>
export function searchIter<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand> & BestOptions,
): IterableIterator<Match<TItem, unknown>> {
  assertOptionKeys(options, BEST_OPTION_KEYS, 'searchIter')
  const threshold = optionalThreshold(options.threshold)
  const scorer = options.scorer
  const normalize = options.normalize
  assertCollection(items)
  const compilation = scorerCompilation(scorer)
  return iterateMatches(
    query,
    items,
    compilation,
    choiceReader(
      stableOptionsOf(options, normalize),
      compilation.prepareChoice,
      compilation.preparedChoiceKey,
      false,
    ),
    normalize,
    threshold,
  )
}

function* iterateMatches<TItem>(
  query: MaybeSequence,
  items: Items<TItem>,
  compilation: AnyMetricCompilation<Direction>,
  choices: ChoiceReader<TItem>,
  normalize: Normalizer | undefined,
  threshold: number | null,
): IterableIterator<Match<TItem, unknown>> {
  const normalized = normalizeQuery(query, normalize)
  const arrayItems = arrayItemsOf(items)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!passesThreshold('similarity', score, threshold)) return
    for (const { item, key } of presentEntries(items, choices)) {
      yield { item, key, score }
    }
    return
  }

  if (impossibleThreshold(compilation, threshold)) return
  const activeThreshold = kernelThreshold(compilation, threshold)
  const sequences = choices.sequences
  if (arrayItems !== null && sequences !== null) {
    let key = 0
    let scored = 0
    for (; key < arrayItems.length && scored < STREAM_PREPARE_AFTER; key++) {
      const item = arrayItems[key]
      const sequence = sequences(item)
      if (sequence === null) continue
      scored++
      const score = compilation.rawScore(normalized, sequence, activeThreshold)
      if (passesThreshold(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    if (key === arrayItems.length) return
    const prepared = compilation.prepareQuery(normalized)
    for (; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const sequence = sequences(item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), activeThreshold)
      if (passesThreshold(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    return
  }

  const prepared: PreparedKernel = compilation.prepareQuery(normalized)
  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      const score = prepared(choice, activeThreshold)
      if (passesThreshold(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    return
  }
  for (const entry of collectionEntries(items)) {
    const choice = choices.read(entry.item)
    if (choice === null) continue
    const score = prepared(choice, activeThreshold)
    if (passesThreshold(compilation.direction, score, threshold)) {
      yield { item: entry.item, key: entry.key, score }
    }
  }
}
