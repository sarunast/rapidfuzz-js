import { assertOptionKeys } from '../core/options.js'
import type { MetricCompilation, PreparedKernel } from '../core/protocol.js'
import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence, Normalizer } from '../core/types.js'
import { assertCollection, collectionEntries } from './collection.js'
import { pushHeap, replaceHeapRoot } from './internal/heap.js'
import type { Match, ScoredEntry } from './results.js'
import {
  BEST_OPTION_KEYS,
  SEARCH_OPTION_KEYS,
  choiceReader,
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  type ChoiceReader,
} from './snapshot.js'
import type {
  BestOptions,
  ItemIterable,
  Items,
  AnyMatcherOptions,
  ResolvedMatcherOptions,
  SearchOptions,
} from './types.js'

/**
 * Every option read exactly once, as `createMatcher` does it.
 *
 * The reader and the query have to be handed the same normalizer: an accessor
 * that answers one function to each passes the prepared-choice check and then
 * scores against a query normalized some other way, which is the silent
 * mismatch that check exists to refuse.
 */
function stableOptionsOf<TItem, TDirection extends Direction, TBrand>(
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
  scorer: ResolvedMatcherOptions<TItem, Direction, TBrand>['scorer'],
  normalize: Normalizer | undefined,
): ResolvedMatcherOptions<TItem, Direction, TBrand> {
  return {
    scorer,
    getText: options.getText,
    getPrepared: options.getPrepared,
    normalize,
    missingItems: options.missingItems,
  }
}

function better(direction: Direction, score: number, current: number): boolean {
  return direction === 'similarity' ? score > current : score < current
}

function qualifies(
  direction: Direction,
  score: number,
  threshold: number | null,
): boolean {
  return (
    threshold === null ||
    (direction === 'similarity' ? score >= threshold : score <= threshold)
  )
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

function arrayItemsOf<TItem>(items: Items<TItem>): readonly TItem[] | null {
  return Array.isArray(items) ? items : null
}

// Array `searchIter` callers often stop after only a handful of matches.
// Preparing a query before the first candidate made that case slower than
// direct pair scoring; after eight scored choices the held representation
// amortizes. Only the array branch adapts: a generic iterable prepares once
// up front, because counting a first-N window over a source that may not
// finish is a different question than this one.
const STREAM_PREPARE_AFTER = 8

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
  // Argument shape is checked before any semantic exit: an impossible
  // threshold must not turn an invalid collection into an empty result.
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
function bestOfCollection<TItem, TDirection extends Direction, TBrand>(
  query: MaybeSequence,
  items: Items<TItem>,
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
  threshold: number | null,
): Match<TItem, unknown> | undefined {
  const scorer = options.scorer
  const compilation = scorerCompilation(scorer)
  const normalize = options.normalize
  const stableOptions = stableOptionsOf(options, scorer, normalize)
  // Before the query is normalized, so `search` at any limit refuses a wrong
  // option in the same order — `limit: 1` delegates here.
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
    if (!qualifies('similarity', score, threshold)) return undefined
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (choices.present(item)) {
          return { item, key, score }
        }
      }
      return undefined
    }
    for (const entry of collectionEntries(items)) {
      if (choices.present(entry.item)) {
        return { item: entry.item, key: entry.key, score }
      }
    }
    return undefined
  }

  if (
    compilation.trusted &&
    impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
  ) {
    return undefined
  }
  const activeThreshold = compilation.trusted
    ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
    : threshold

  const prepared = compilation.prepareQuery(normalized)
  const optimal = compilation.trusted
    ? compilation.direction === 'similarity'
      ? compilation.bounds[1]
      : compilation.bounds[0]
    : null
  let found: Match<TItem, unknown> | undefined
  let cutoff = activeThreshold

  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      if (choice === null) continue
      const score = prepared(choice, cutoff)
      if (!qualifies(compilation.direction, score, activeThreshold)) continue
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
    if (!qualifies(compilation.direction, score, activeThreshold)) continue
    if (found === undefined || better(compilation.direction, score, found.score)) {
      found = { item: entry.item, key: entry.key, score }
      cutoff = score
      if (optimal !== null && score === optimal) break
    }
  }
  return found
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
 * Use {@link searchIter} when you want to stop early instead of ranking
 * everything, and {@link bestMatch} when one winner is enough.
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
  const stableOptions = stableOptionsOf(options, scorer, normalize)
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
    if (!qualifies('similarity', score, threshold)) return []
    const results: Match<TItem, unknown>[] = []
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (choices.present(item)) {
          results.push({ item, key, score })
          if (limit !== null && results.length === limit) break
        }
      }
      return results
    }
    for (const entry of collectionEntries(items)) {
      if (choices.present(entry.item)) {
        results.push({ item: entry.item, key: entry.key, score })
        if (limit !== null && results.length === limit) break
      }
    }
    return results
  }

  if (
    compilation.trusted &&
    impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
  ) {
    return []
  }
  const activeThreshold = compilation.trusted
    ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
    : threshold

  const prepared = compilation.prepareQuery(normalized)
  const results: ScoredEntry<TItem, unknown>[] = []
  const heapWorse = (
    left: ScoredEntry<TItem, unknown>,
    right: ScoredEntry<TItem, unknown>,
  ) => worse(compilation.direction, left, right)
  // Once a full heap holds nothing but optimal scores, later candidates can
  // only tie, and a tie loses on order — so the scan is finished. The Matcher
  // drivers stop on the same condition.
  const optimal = compilation.trusted
    ? compilation.direction === 'similarity'
      ? compilation.bounds[1]
      : compilation.bounds[0]
    : null
  let cutoff = activeThreshold

  if (arrayItems !== null) {
    // An array index is already the source order the heap breaks ties on, so
    // the counter the generic branch keeps is one the array branch can read.
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      if (choice === null) continue
      const score = prepared(choice, cutoff)
      if (qualifies(compilation.direction, score, activeThreshold)) {
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
      if (qualifies(compilation.direction, score, activeThreshold)) {
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
 * Use {@link search} when you need results ranked, and {@link bestMatch} when
 * you want the highest-scoring one rather than the first acceptable one.
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
  // Call options and collection shape are read and checked here, so a caller
  // who mutates their options object before iterating cannot change a search
  // already asked for, and a wrong threshold, scorer, collection or
  // `missingItems` is refused at the call rather than on the first `next()`.
  // The query is processed lazily with the scoring — that is what the
  // iterator is for, so an invalid query still throws from `next()`.
  assertOptionKeys(options, BEST_OPTION_KEYS, 'searchIter')
  const threshold = optionalThreshold(options.threshold)
  // Read before the generator exists, each exactly once: what the iterator
  // scores with is settled at the call, not at the first `next()`.
  const scorer = options.scorer
  const normalize = options.normalize
  assertCollection(items)
  const compilation = scorerCompilation(scorer)
  return iterateMatches(
    query,
    items,
    compilation,
    choiceReader(
      stableOptionsOf(options, scorer, normalize),
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
  compilation: MetricCompilation<Direction>,
  choices: ChoiceReader<TItem>,
  normalize: Normalizer | undefined,
  threshold: number | null,
): IterableIterator<Match<TItem, unknown>> {
  const normalized = normalizeQuery(query, normalize)
  const arrayItems = arrayItemsOf(items)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!qualifies('similarity', score, threshold)) return
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (choices.present(item)) yield { item, key, score }
      }
      return
    }
    for (const entry of collectionEntries(items)) {
      if (choices.present(entry.item)) {
        yield { item: entry.item, key: entry.key, score }
      }
    }
    return
  }

  if (
    compilation.trusted &&
    impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
  ) {
    return
  }
  const activeThreshold = compilation.trusted
    ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
    : threshold
  const sequences = choices.sequences
  // The raw-score window scores sequences directly, so it belongs to text mode
  // alone: a prepared choice is not one, and prepared mode has a held
  // representation to amortize from the first candidate anyway.
  if (arrayItems !== null && sequences !== null) {
    let key = 0
    let scored = 0
    for (; key < arrayItems.length && scored < STREAM_PREPARE_AFTER; key++) {
      const item = arrayItems[key]
      const sequence = sequences(item)
      if (sequence === null) continue
      scored++
      const score = compilation.rawScore(normalized, sequence, activeThreshold)
      if (qualifies(compilation.direction, score, threshold)) {
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
      if (qualifies(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    return
  }

  const prepared: PreparedKernel = compilation.prepareQuery(normalized)
  // Only prepared mode reaches this array loop — text mode took the window
  // above — and a prepared choice is never skipped: a missing or foreign one
  // throws where it is read.
  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const choice = choices.read(item)
      const score = prepared(choice, activeThreshold)
      if (qualifies(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    return
  }
  for (const entry of collectionEntries(items)) {
    const choice = choices.read(entry.item)
    if (choice === null) continue
    const score = prepared(choice, activeThreshold)
    if (qualifies(compilation.direction, score, threshold)) {
      yield { item: entry.item, key: entry.key, score }
    }
  }
}
