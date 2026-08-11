import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence } from '../core/types.js'
import { collectionEntries } from './collection.js'
import { bestDistance } from './internal/bestDistance.js'
import { bestSimilarity } from './internal/bestSimilarity.js'
import { topDistance } from './internal/topDistance.js'
import { topSimilarity } from './internal/topSimilarity.js'
import type { StoredItem } from './internal/types.js'
import type { Match } from './results.js'
import {
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  sequenceReader,
} from './snapshot.js'
import type {
  BestOptions,
  Items,
  Matcher,
  MatcherOptions,
  SearchIterOptions,
  SearchOptions,
} from './types.js'

function missingBest<T, K>(
  items: readonly StoredItem<T, K>[],
  score: number,
  threshold: number | null,
): Match<T, K> | undefined {
  if (threshold !== null && score < threshold) return undefined
  const first = items[0]
  return first === undefined ? undefined : { item: first.item, key: first.key, score }
}

function missingTop<T, K>(
  items: readonly StoredItem<T, K>[],
  score: number,
  threshold: number | null,
  limit: number | null,
): readonly Match<T, K>[] {
  if (threshold !== null && score < threshold) return []
  const selected = limit === null ? items : items.slice(0, limit)
  return selected.map(({ item, key }) => ({ item, key, score }))
}

export function createMatcher<T, D extends Direction>(
  items: readonly T[],
  options: MatcherOptions<T, D>,
): Matcher<T, number, D>
export function createMatcher<K, T, D extends Direction>(
  items: ReadonlyMap<K, T>,
  options: MatcherOptions<T, D>,
): Matcher<T, K, D>
export function createMatcher<T, D extends Direction>(
  items: Iterable<T>,
  options: MatcherOptions<T, D>,
): Matcher<T, number, D>
export function createMatcher<T, D extends Direction>(
  items: Readonly<Record<string, T>>,
  options: MatcherOptions<T, D>,
): Matcher<T, string, D>
export function createMatcher<T, D extends Direction>(
  items: Items<T>,
  options: MatcherOptions<T, D>,
): Matcher<T, unknown, D> {
  const scorer = options.scorer
  const normalize = options.normalize
  const compilation = scorerCompilation(scorer)
  const stableOptions: MatcherOptions<T, Direction> = {
    scorer: options.scorer,
    ...(options.getText === undefined ? {} : { getText: options.getText }),
    ...(options.normalize === undefined ? {} : { normalize: options.normalize }),
    ...(options.missingItems === undefined ? {} : { missingItems: options.missingItems }),
  }
  const stored: StoredItem<T, unknown>[] = []
  const readSequence = sequenceReader(stableOptions, true)
  if (Array.isArray(items)) {
    for (let key = 0; key < items.length; key++) {
      const item = items[key]
      const sequence = readSequence(item)
      if (sequence !== null) {
        stored.push({ item, key, prepared: compilation.prepareChoice(sequence) })
      }
    }
  } else {
    for (const entry of collectionEntries(items)) {
      const sequence = readSequence(entry.item)
      if (sequence !== null) {
        stored.push({
          item: entry.item,
          key: entry.key,
          prepared: compilation.prepareChoice(sequence),
        })
      }
    }
  }

  const best = (
    query: MaybeSequence,
    call?: BestOptions,
  ): Match<T, unknown> | undefined => {
    const threshold = optionalThreshold(call?.threshold)
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      const missingScore = compilation.score(query, '', threshold)
      return missingBest(stored, missingScore, threshold)
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
    const score = compilation.prepareQuery(normalized)
    const optimal = compilation.trusted
      ? compilation.direction === 'similarity'
        ? compilation.bounds[1]
        : compilation.bounds[0]
      : null
    return compilation.direction === 'similarity'
      ? bestSimilarity(stored, score, activeThreshold, optimal)
      : bestDistance(stored, score, activeThreshold, optimal)
  }
  const search = (
    query: MaybeSequence,
    call?: SearchOptions,
  ): readonly Match<T, unknown>[] => {
    const limit = resultLimit(call?.limit)
    if (limit === 0) return []
    const threshold = optionalThreshold(call?.threshold)
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      const missingScore = compilation.score(query, '', threshold)
      return missingTop(stored, missingScore, threshold, limit)
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
    const score = compilation.prepareQuery(normalized)
    const optimal = compilation.trusted
      ? compilation.direction === 'similarity'
        ? compilation.bounds[1]
        : compilation.bounds[0]
      : null
    return compilation.direction === 'similarity'
      ? topSimilarity(stored, score, activeThreshold, limit, optimal)
      : topDistance(stored, score, activeThreshold, limit, optimal)
  }
  const searchIter = (
    query: MaybeSequence,
    call?: SearchIterOptions,
  ): IterableIterator<Match<T, unknown>> => {
    function* iterate(): Generator<Match<T, unknown>> {
      const threshold = optionalThreshold(call?.threshold)
      const normalized = normalizeQuery(query, normalize)
      if (normalized === null) {
        const missingScore = compilation.score(query, '', threshold)
        if (threshold !== null && missingScore < threshold) return
        for (let index = 0; index < stored.length; index++) {
          const entry = stored[index]
          yield { item: entry.item, key: entry.key, score: missingScore }
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
      const score = compilation.prepareQuery(normalized)
      for (let index = 0; index < stored.length; index++) {
        const entry = stored[index]
        const value = score(entry.prepared, activeThreshold)
        if (
          threshold === null ||
          (compilation.direction === 'similarity'
            ? value >= threshold
            : value <= threshold)
        ) {
          yield { item: entry.item, key: entry.key, score: value }
        }
      }
    }

    return iterate()
  }
  return Object.freeze({
    size: stored.length,
    scorer,
    best,
    search,
    searchIter,
  })
}
