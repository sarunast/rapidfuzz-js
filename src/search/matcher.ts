import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { Direction, MaybeSequence } from '../core/types.js'
import { assertCollection, collectionEntries } from './collection.js'
import { bestDistance } from './internal/bestDistance.js'
import { bestSimilarity } from './internal/bestSimilarity.js'
import { topDistance } from './internal/topDistance.js'
import { topSimilarity } from './internal/topSimilarity.js'
import type { StoredItem } from './internal/types.js'
import type { Match } from './results.js'
import {
  choiceReader,
  normalizeQuery,
  optionalThreshold,
  resultLimit,
} from './snapshot.js'
import type {
  AnyMatcherOptions,
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
function missingSimilarityBest<T, K>(
  items: readonly StoredItem<T, K>[],
  score: number,
  threshold: number | null,
): Match<T, K> | undefined {
  if (threshold !== null && score < threshold) return undefined
  const first = items[0]
  return first === undefined ? undefined : { item: first.item, key: first.key, score }
}

function missingSimilarityTop<T, K>(
  items: readonly StoredItem<T, K>[],
  score: number,
  threshold: number | null,
  limit: number | null,
): readonly Match<T, K>[] {
  if (threshold !== null && score < threshold) return []
  const length = limit === null ? items.length : Math.min(items.length, limit)
  const matches: Match<T, K>[] = new Array(length)
  for (let index = 0; index < length; index++) {
    const entry = items[index]
    matches[index] = { item: entry.item, key: entry.key, score }
  }
  return matches
}

export function createMatcher<T, D extends Direction, B>(
  items: readonly T[],
  options: AnyMatcherOptions<T, D, B>,
): Matcher<T, number, D>
export function createMatcher<K, T, D extends Direction, B>(
  items: ReadonlyMap<K, T>,
  options: AnyMatcherOptions<T, D, B>,
): Matcher<T, K, D>
export function createMatcher<T, D extends Direction, B>(
  items: ItemIterable<T>,
  options: AnyMatcherOptions<T, D, B>,
): Matcher<T, number, D>
export function createMatcher<T, D extends Direction, B>(
  items: Readonly<Record<string, T>>,
  options: AnyMatcherOptions<T, D, B>,
): Matcher<T, string, D>
export function createMatcher<T, D extends Direction, B>(
  items: Items<T>,
  options: AnyMatcherOptions<T, D, B>,
): Matcher<T, unknown, D> {
  const scorer = options.scorer
  const normalize = options.normalize
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
  const stableOptions: AnyMatcherOptions<T, Direction, B> =
    options.getPrepared === undefined
      ? {
          scorer: options.scorer,
          getText: options.getText,
          normalize: options.normalize,
          missingItems: options.missingItems,
        }
      : {
          scorer: options.scorer,
          getPrepared: options.getPrepared,
          normalize: options.normalize,
          // Copied although the type says they are absent, so a JavaScript
          // caller who passed both kinds of accessor is refused here rather
          // than quietly served the prepared one.
          getText: options.getText,
          missingItems: options.missingItems,
        }
  const stored: StoredItem<T, unknown>[] = []
  // Every handle is resolved here, once, so a query pays nothing for the mode
  // it was built in — the drivers read `prepared` the same way either way.
  const choices = choiceReader(
    stableOptions,
    compilation.prepareChoice,
    compilation.preparedChoiceKey,
    true,
  )
  if (Array.isArray(items)) {
    for (let key = 0; key < items.length; key++) {
      const item = items[key]
      const prepared = choices.read(item)
      if (prepared !== null) {
        stored.push({ item, key, prepared })
      }
    }
  } else {
    for (const entry of collectionEntries(items)) {
      const prepared = choices.read(entry.item)
      if (prepared !== null) {
        stored.push({ item: entry.item, key: entry.key, prepared })
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
      return missingSimilarityBest(stored, missingScore, threshold)
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
    return direction === 'similarity'
      ? bestSimilarity(stored, score, activeThreshold, optimal)
      : bestDistance(stored, score, activeThreshold, optimal)
  }
  const search = (
    query: MaybeSequence,
    call?: SearchOptions,
  ): readonly Match<T, unknown>[] => {
    const limit = resultLimit(call?.limit)
    const threshold = optionalThreshold(call?.threshold)
    if (limit === 0) return []
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      const missingScore = compilation.score(query, '', threshold)
      return missingSimilarityTop(stored, missingScore, threshold, limit)
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
    return direction === 'similarity'
      ? topSimilarity(stored, score, activeThreshold, limit, optimal)
      : topDistance(stored, score, activeThreshold, limit, optimal)
  }
  const searchIter = (
    query: MaybeSequence,
    call?: BestOptions,
  ): IterableIterator<Match<T, unknown>> => {
    // Read where the call is made, not where iteration starts: a caller who
    // mutates their options object between the two would otherwise change a
    // search already asked for. Scoring stays lazy; only the number is taken.
    const threshold = optionalThreshold(call?.threshold)
    function* iterate(): Generator<Match<T, unknown>> {
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
        impossibleTrustedThreshold(direction, compilation.bounds, threshold)
      ) {
        return
      }
      const activeThreshold = compilation.trusted
        ? trustedKernelThreshold(direction, compilation.bounds, threshold)
        : threshold
      const score = compilation.prepareQuery(normalized)
      const similarity = direction === 'similarity'
      for (let index = 0; index < stored.length; index++) {
        const entry = stored[index]
        const value = score(entry.prepared, activeThreshold)
        if (
          threshold === null ||
          (similarity ? value >= threshold : value <= threshold)
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
