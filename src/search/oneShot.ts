import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence } from '../core/types.js'
import { collectionEntries } from './collection.js'
import { pushHeap, replaceHeapRoot } from './internal/heap.js'
import type { Match, ScoredEntry } from './results.js'
import {
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  sequenceReader,
} from './snapshot.js'
import type { BestOptions, Items, MatcherOptions, SearchOptions } from './types.js'

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
  items: Iterable<T>,
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
  const threshold = optionalThreshold(options.threshold)
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
  items: Iterable<T>,
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
  const limit = resultLimit(options.limit)
  if (limit === 0) return []
  if (limit === 1) {
    const match = bestMatch(query, items, options)
    return match === undefined ? [] : [match]
  }
  const threshold = optionalThreshold(options.threshold)
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
  let cutoff = activeThreshold
  let order = 0

  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      const sequence = readSequence(item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), cutoff)
      if (qualifies(compilation.direction, score, activeThreshold)) {
        if (limit === null) {
          results.push({ item, key, score, order })
        } else if (results.length < limit) {
          pushHeap(results, { item, key, score, order }, heapWorse)
          if (results.length === limit) cutoff = results[0].score
        } else if (better(compilation.direction, score, results[0].score)) {
          replaceHeapRoot(results, { item, key, score, order }, heapWorse)
          cutoff = results[0].score
        }
      }
      order++
    }
  } else {
    for (const entry of collectionEntries(items)) {
      const sequence = readSequence(entry.item)
      if (sequence === null) continue
      const score = prepared(compilation.prepareChoice(sequence), cutoff)
      if (qualifies(compilation.direction, score, activeThreshold)) {
        if (limit === null) {
          results.push({ item: entry.item, key: entry.key, score, order })
        } else if (results.length < limit) {
          pushHeap(results, { item: entry.item, key: entry.key, score, order }, heapWorse)
          if (results.length === limit) cutoff = results[0].score
        } else if (better(compilation.direction, score, results[0].score)) {
          replaceHeapRoot(
            results,
            { item: entry.item, key: entry.key, score, order },
            heapWorse,
          )
          cutoff = results[0].score
        }
      }
      order++
    }
  }
  return orderedResults(compilation.direction, results)
}
