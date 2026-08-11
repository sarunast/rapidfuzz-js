import { scorerCompilation } from '../core/scorer.js'
import type { Direction, MaybeSequence } from '../core/types.js'
import { collectionEntries } from './collection.js'
import type { Match } from './results.js'
import {
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  searchableSequence,
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

export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, unknown> | undefined {
  const threshold = optionalThreshold(options.threshold)
  const compilation = scorerCompilation(options.scorer)
  const normalized = normalizeQuery(query, options.normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  const optimal = compilation.trusted
    ? compilation.direction === 'similarity'
      ? compilation.bounds[1]
      : compilation.bounds[0]
    : null
  const stableOptions: MatcherOptions<T, Direction> = options
  let found: Match<T, unknown> | undefined
  let cutoff = threshold
  for (const entry of collectionEntries(items)) {
    const sequence = searchableSequence(entry.item, stableOptions, false)
    if (sequence === null) continue
    const score =
      prepared === null
        ? compilation.score(query, sequence, cutoff)
        : prepared(compilation.prepareChoice(sequence), cutoff)
    if (!qualifies(compilation.direction, score, threshold)) continue
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
  items: Items<T>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, unknown>[] {
  const limit = resultLimit(options.limit)
  if (limit === 0) return []
  const threshold = optionalThreshold(options.threshold)
  const compilation = scorerCompilation(options.scorer)
  const normalized = normalizeQuery(query, options.normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  const stableOptions: MatcherOptions<T, Direction> = options
  const results: Array<Match<T, unknown> & { readonly order: number }> = []
  let order = 0
  for (const entry of collectionEntries(items)) {
    const sequence = searchableSequence(entry.item, stableOptions, false)
    if (sequence === null) continue
    const score =
      prepared === null
        ? compilation.score(query, sequence, threshold)
        : prepared(compilation.prepareChoice(sequence), threshold)
    if (qualifies(compilation.direction, score, threshold)) {
      results.push({ item: entry.item, key: entry.key, score, order })
    }
    order++
  }
  results.sort((a, b) => {
    const byScore =
      compilation.direction === 'similarity' ? b.score - a.score : a.score - b.score
    return byScore || a.order - b.order
  })
  const selected = limit === null ? results : results.slice(0, limit)
  return selected.map(({ item, key, score }) => ({ item, key, score }))
}
