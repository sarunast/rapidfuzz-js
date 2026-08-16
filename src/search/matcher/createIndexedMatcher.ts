import { assertOptionKeys } from '#core/options.js'
import type { ChoiceIndex, SelectedChoices } from '#core/scoring/choiceIndex.js'
import { scorerCompilation } from '#core/scoring/scorer.js'
import {
  impossibleThreshold,
  kernelThreshold,
  optionalThreshold,
} from '#core/scoring/threshold.js'
import type { MaybeSequence } from '#core/types.js'

import type { Match } from '../results.js'
import { assertCollection } from '../shared/collection.js'
import {
  CALL_BEST_KEYS,
  CALL_SEARCH_KEYS,
  INDEXED_MATCHER_OPTION_KEYS,
  resultLimit,
} from '../shared/options.js'
import { normalizeQuery, sequenceReader } from '../shared/readers.js'
import type {
  BestOptions,
  IndexedMatcherOptions,
  ItemIterable,
  Items,
  Matcher,
  SearchOptions,
} from '../types.js'
import { buildChoiceTable, matchAt } from './choiceTable.js'
import {
  missingSimilarityBest,
  missingSimilarityMatches,
  missingSimilarityTop,
} from './missingQuery.js'

/**
 * Prepare a collection into one searchable index and query it many times.
 *
 * The same {@link Matcher} `createMatcher` returns, over a different
 * representation: instead of one prepared handle per choice, the whole
 * collection becomes a single inverted structure that answers a query without
 * scoring choices one at a time. Every member behaves identically — same
 * scores, same order, same ties, same `key` — so swapping the constructor is
 * the whole change at a call site.
 *
 * ```ts
 * const matcher = createIndexedMatcher(files, {
 *   scorer: createScorer(diceSimilarity, { gramSize: 3 }),
 *   getText: (file) => file.path,
 * })
 * matcher.search('src/algorthms/dice.ts', { limit: 5, threshold: 0.5 })
 * ```
 *
 * On 10,000 `node_modules` file paths at `gramSize: 3`, searching for a path
 * that exists or one with a typo in it: **11–13x faster** per query. Construction
 * retained **256 bytes a choice against 1,282** — 5x less. Query scratch is
 * separate from that figure and reused between queries: a broad query reserves a
 * result slot per choice, and an oversized reservation is released once query
 * demand returns to the normal retained range.
 *
 * Three things to weigh before reaching for it:
 *
 * - **It is not uniformly faster, and can be slower.** The win comes from a
 *   query's grams naming few choices. On that same corpus a query of
 *   `'node_modules/'` — grams nearly every choice shares — measured **0.7x**,
 *   slower than scoring every choice, and a two-letter alphabet loses
 *   outright.
 * - **Construction costs more**, about 1.2x, where a prepared collection packs
 *   each choice's grams into two typed arrays and the index has a corpus-wide
 *   structure to assemble. It is a per-query win paid for up front.
 * - **`searchIter` settles which choices qualify before yielding the first**,
 *   where `createMatcher` scores lazily. Same values, same order; a caller who
 *   breaks out early saves the scoring it would have skipped there, and only
 *   the cost of building the results it never asked for.
 *
 * @param items Array, `Map`, plain object or any iterable — the shape decides
 * what `key` is on every result, exactly as it does for `createMatcher`.
 * @returns A frozen {@link Matcher} exposing `best`, `search`, `searchIter`,
 * `size` and the `scorer` it was built from.
 * @throws `TypeError` for an unknown option key — `getPrepared` among them,
 * since a prepared handle is the representation this replaces — for a scorer
 * with no indexed representation, for a gap when `missingItems: 'throw'`, or
 * for a choice whose elements are not integers.
 */
export function createIndexedMatcher<TItem, TBrand>(
  items: readonly TItem[],
  options: IndexedMatcherOptions<TItem, TBrand>,
): Matcher<TItem, number, 'similarity', TBrand>
export function createIndexedMatcher<TKey, TItem, TBrand>(
  items: ReadonlyMap<TKey, TItem>,
  options: IndexedMatcherOptions<TItem, TBrand>,
): Matcher<TItem, TKey, 'similarity', TBrand>
export function createIndexedMatcher<TItem, TBrand>(
  items: ItemIterable<TItem>,
  options: IndexedMatcherOptions<TItem, TBrand>,
): Matcher<TItem, number, 'similarity', TBrand>
export function createIndexedMatcher<TItem, TBrand>(
  items: Readonly<Record<string, TItem>>,
  options: IndexedMatcherOptions<TItem, TBrand>,
): Matcher<TItem, string, 'similarity', TBrand>
export function createIndexedMatcher<TItem, TBrand>(
  items: Items<TItem>,
  options: IndexedMatcherOptions<TItem, TBrand>,
): Matcher<TItem, unknown, 'similarity', TBrand> {
  assertOptionKeys(options, INDEXED_MATCHER_OPTION_KEYS, 'createIndexedMatcher')
  const scorer = options.scorer
  const normalize = options.normalize
  const getText = options.getText
  const missingItems = options.missingItems
  assertCollection(items)
  const compilation = scorerCompilation(scorer)
  const indexChoices = compilation.indexChoices
  if (indexChoices === undefined) {
    throw new TypeError(
      'createIndexedMatcher: this scorer has no indexed representation. ' +
        'Indexed search is available for dice.similarity, cosine.similarity ' +
        'and tversky.similarity.',
    )
  }
  const read = sequenceReader({ getText, normalize, missingItems }, false)
  const builder = indexChoices()
  const table = buildChoiceTable(items, (item) => {
    const sequence = read(item)
    if (sequence === null) return false
    builder.add(sequence)
    return true
  })
  const index: ChoiceIndex = builder.seal()

  /** What a query with no text to score is worth against every choice. */
  const missingScoreOf = (query: MaybeSequence, threshold: number | null): number =>
    compilation.score(query, '', threshold)

  const materialize = (found: SelectedChoices): Match<TItem, unknown>[] => {
    const matches: Match<TItem, unknown>[] = new Array(found.length)
    for (let at = 0; at < found.length; at++) {
      matches[at] = matchAt(table, found.ids[at], found.scores[at])
    }
    return matches
  }

  const best = (
    query: MaybeSequence,
    call?: BestOptions,
  ): Match<TItem, unknown> | undefined => {
    if (call !== undefined) assertOptionKeys(call, CALL_BEST_KEYS, 'matcher.best')
    const threshold = optionalThreshold(call?.threshold)
    const normalized = normalizeQuery(query, normalize)
    if (normalized === null) {
      return missingSimilarityBest(table, missingScoreOf(query, threshold), threshold)
    }
    if (impossibleThreshold(compilation, threshold)) return undefined
    const found = index.select(normalized, kernelThreshold(compilation, threshold), 1)
    return found.length === 0 ? undefined : matchAt(table, found.ids[0], found.scores[0])
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
      const score = missingScoreOf(query, threshold)
      return missingSimilarityTop(table, score, threshold, limit)
    }
    if (impossibleThreshold(compilation, threshold)) return []
    return materialize(
      index.select(normalized, kernelThreshold(compilation, threshold), limit),
    )
  }

  const searchIter = (
    query: MaybeSequence,
    call?: BestOptions,
  ): IterableIterator<Match<TItem, unknown>> => {
    if (call !== undefined) assertOptionKeys(call, CALL_BEST_KEYS, 'matcher.searchIter')
    const threshold = optionalThreshold(call?.threshold)
    function* iterate(): Generator<Match<TItem, unknown>> {
      const normalized = normalizeQuery(query, normalize)
      if (normalized === null) {
        const score = missingScoreOf(query, threshold)
        yield* missingSimilarityMatches(table, score, threshold)
        return
      }
      if (impossibleThreshold(compilation, threshold)) return
      const found = index.scan(normalized, kernelThreshold(compilation, threshold))
      const ids = found.ids.slice(0, found.length)
      const scores = found.scores.slice(0, found.length)
      for (let at = 0; at < ids.length; at++) yield matchAt(table, ids[at], scores[at])
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
