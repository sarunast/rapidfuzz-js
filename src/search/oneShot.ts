import type { MetricCompilation, PreparedKernel } from '../core/protocol.js'
import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence, Normalizer } from '../core/types.js'
import { assertCollection, collectionEntries } from './collection.js'
import { pushHeap, replaceHeapRoot } from './internal/heap.js'
import type { Match, ScoredEntry } from './results.js'
import {
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  sequenceReader,
  type SequenceReader,
} from './snapshot.js'
import type {
  BestOptions,
  ItemIterable,
  Items,
  MatcherOptions,
  SearchOptions,
} from './types.js'

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

function worse<T, K>(
  direction: Direction,
  left: ScoredEntry<T, K>,
  right: ScoredEntry<T, K>,
): boolean {
  if (left.score !== right.score) return better(direction, right.score, left.score)
  return left.order > right.order
}

function orderedResults<T, K>(
  direction: Direction,
  entries: ScoredEntry<T, K>[],
): readonly Match<T, K>[] {
  entries.sort((a, b) => {
    const byScore = direction === 'similarity' ? b.score - a.score : a.score - b.score
    return byScore || a.order - b.order
  })
  return entries.map(({ item, key, score }) => ({ item, key, score }))
}

function arrayItemsOf<T>(items: Items<T>): readonly T[] | null {
  return Array.isArray(items) ? items : null
}

// Array `searchIter` callers often stop after only a handful of matches.
// Preparing a query before the first candidate made that case slower than
// direct pair scoring; after eight scored choices the held representation
// amortizes. Only the array branch adapts: a generic iterable prepares once
// up front, because counting a first-N window over a source that may not
// finish is a different question than this one.
const STREAM_PREPARE_AFTER = 8

export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: readonly T[],
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, number> | undefined
export function bestMatch<K, T, D extends Direction>(
  query: MaybeSequence,
  items: ReadonlyMap<K, T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, K> | undefined
export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: ItemIterable<T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, number> | undefined
export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: Readonly<Record<string, T>>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, string> | undefined
export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, unknown> | undefined
export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, unknown> | undefined {
  // Argument shape is checked before any semantic exit: an impossible
  // threshold must not turn an invalid collection into an empty result.
  const threshold = optionalThreshold(options.threshold)
  assertCollection(items)
  const compilation = scorerCompilation(options.scorer)
  const normalized = normalizeQuery(query, options.normalize)
  const stableOptions: MatcherOptions<T, Direction> = options
  const arrayItems = arrayItemsOf(items)
  const readSequence = sequenceReader(stableOptions, false)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!qualifies('similarity', score, threshold)) return undefined
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (readSequence(item) !== null) {
          return { item, key, score }
        }
      }
      return undefined
    }
    for (const entry of collectionEntries(items)) {
      if (readSequence(entry.item) !== null) {
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
  let found: Match<T, unknown> | undefined
  let cutoff = activeThreshold

  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const sequence = readSequence(item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), cutoff)
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
    const sequence = readSequence(entry.item)
    if (sequence === null) continue
    const score = prepared(compilation.prepareChoice(sequence), cutoff)
    if (!qualifies(compilation.direction, score, activeThreshold)) continue
    if (found === undefined || better(compilation.direction, score, found.score)) {
      found = { item: entry.item, key: entry.key, score }
      cutoff = score
      if (optimal !== null && score === optimal) break
    }
  }
  return found
}

export function search<T, D extends Direction>(
  query: MaybeSequence,
  items: readonly T[],
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, number>[]
export function search<K, T, D extends Direction>(
  query: MaybeSequence,
  items: ReadonlyMap<K, T>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, K>[]
export function search<T, D extends Direction>(
  query: MaybeSequence,
  items: ItemIterable<T>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, number>[]
export function search<T, D extends Direction>(
  query: MaybeSequence,
  items: Readonly<Record<string, T>>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, string>[]
export function search<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, unknown>[] {
  // Argument shape is checked before any semantic exit: `limit: 0` must not
  // excuse an invalid collection or a non-finite threshold.
  const limit = resultLimit(options.limit)
  const threshold = optionalThreshold(options.threshold)
  assertCollection(items)
  if (limit === 0) return []
  if (limit === 1) {
    const match = bestMatch(query, items, options)
    return match === undefined ? [] : [match]
  }
  const compilation = scorerCompilation(options.scorer)
  const normalized = normalizeQuery(query, options.normalize)
  const stableOptions: MatcherOptions<T, Direction> = options
  const arrayItems = arrayItemsOf(items)
  const readSequence = sequenceReader(stableOptions, false)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!qualifies('similarity', score, threshold)) return []
    const results: Match<T, unknown>[] = []
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (readSequence(item) !== null) {
          results.push({ item, key, score })
          if (limit !== null && results.length === limit) break
        }
      }
      return results
    }
    for (const entry of collectionEntries(items)) {
      if (readSequence(entry.item) !== null) {
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
  const results: ScoredEntry<T, unknown>[] = []
  const heapWorse = (left: ScoredEntry<T, unknown>, right: ScoredEntry<T, unknown>) =>
    worse(compilation.direction, left, right)
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
      const sequence = readSequence(item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), cutoff)
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
      const sequence = readSequence(entry.item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), cutoff)
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

export function searchIter<T, D extends Direction>(
  query: MaybeSequence,
  items: readonly T[],
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, number>>
export function searchIter<K, T, D extends Direction>(
  query: MaybeSequence,
  items: ReadonlyMap<K, T>,
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, K>>
export function searchIter<T, D extends Direction>(
  query: MaybeSequence,
  items: ItemIterable<T>,
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, number>>
export function searchIter<T, D extends Direction>(
  query: MaybeSequence,
  items: Readonly<Record<string, T>>,
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, string>>
export function searchIter<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, unknown>>
export function searchIter<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): IterableIterator<Match<T, unknown>> {
  // Call options and collection shape are read and checked here, so a caller
  // who mutates their options object before iterating cannot change a search
  // already asked for, and a wrong threshold, scorer, collection or
  // `missingItems` is refused at the call rather than on the first `next()`.
  // The query is processed lazily with the scoring — that is what the
  // iterator is for, so an invalid query still throws from `next()`.
  const threshold = optionalThreshold(options.threshold)
  assertCollection(items)
  const stableOptions: MatcherOptions<T, Direction> = options
  return iterateMatches(
    query,
    items,
    scorerCompilation(options.scorer),
    sequenceReader(stableOptions, false),
    options.normalize,
    threshold,
  )
}

function* iterateMatches<T>(
  query: MaybeSequence,
  items: Items<T>,
  compilation: MetricCompilation<Direction>,
  readSequence: SequenceReader<T>,
  normalize: Normalizer | undefined,
  threshold: number | null,
): IterableIterator<Match<T, unknown>> {
  const normalized = normalizeQuery(query, normalize)
  const arrayItems = arrayItemsOf(items)

  if (normalized === null) {
    const score = compilation.score(query, '', threshold)
    if (!qualifies('similarity', score, threshold)) return
    if (arrayItems !== null) {
      for (let key = 0; key < arrayItems.length; key++) {
        const item = arrayItems[key]
        if (readSequence(item) !== null) yield { item, key, score }
      }
      return
    }
    for (const entry of collectionEntries(items)) {
      if (readSequence(entry.item) !== null) {
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
  if (arrayItems !== null) {
    let key = 0
    let scored = 0
    for (; key < arrayItems.length && scored < STREAM_PREPARE_AFTER; key++) {
      const item = arrayItems[key]
      const sequence = readSequence(item)
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
      const sequence = readSequence(item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), activeThreshold)
      if (qualifies(compilation.direction, score, threshold)) {
        yield { item, key, score }
      }
    }
    return
  }

  const prepared: PreparedKernel = compilation.prepareQuery(normalized)
  for (const entry of collectionEntries(items)) {
    const sequence = readSequence(entry.item)
    if (sequence === null) continue
    const score = prepared(compilation.prepareChoice(sequence), activeThreshold)
    if (qualifies(compilation.direction, score, threshold)) {
      yield { item: entry.item, key: entry.key, score }
    }
  }
}
