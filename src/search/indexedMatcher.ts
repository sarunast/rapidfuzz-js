import { assertOptionKeys } from '../core/options.js'
import type { ChoiceIndex, SelectedChoices } from '../core/protocol.js'
import { scorerCompilation } from '../core/scorer.js'
import { impossibleTrustedThreshold, trustedKernelThreshold } from '../core/threshold.js'
import type { MaybeSequence } from '../core/types.js'
import { assertCollection, collectionEntries } from './collection.js'
import type { Match } from './results.js'
import {
  CALL_BEST_KEYS,
  CALL_SEARCH_KEYS,
  INDEXED_MATCHER_OPTION_KEYS,
  normalizeQuery,
  optionalThreshold,
  resultLimit,
  sequenceReader,
} from './snapshot.js'
import type {
  BestOptions,
  IndexedMatcherOptions,
  ItemIterable,
  Items,
  Matcher,
  SearchOptions,
} from './types.js'

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
 * that exists or one with a typo in it: **45–63x faster** per query, retaining
 * **235 bytes a choice against 18,049** — 77x less. Construction is **faster
 * too**, about 0.6x, because the index reads each choice's grams once instead
 * of building a profile per choice and keeping it.
 *
 * Two things to weigh before reaching for it:
 *
 * - **It is not uniformly faster, and can be slower.** The win comes from a
 *   query's grams naming few choices. On that same corpus a query of
 *   `'node_modules/'` — grams nearly every choice shares — measured **1x** for
 *   Dice, and a two-letter alphabet loses outright. Cosine keeps its lead there
 *   because its exhaustive path has no length bound to prune with.
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
  // Read exactly once each, as `createMatcher` does: a getter could otherwise
  // answer one thing to the reader and another to a second look.
  const scorer = options.scorer
  const normalize = options.normalize
  const getText = options.getText
  const missingItems = options.missingItems
  assertCollection(items)
  const compilation = scorerCompilation(scorer)
  const indexChoices = compilation.indexChoices
  // Runtime rather than static, because a custom scorer built from a plain
  // function can never be enumerated in a type, and naming the two built-in
  // metrics in a type would put algorithm identity inside `search/`.
  if (indexChoices === undefined) {
    throw new TypeError(
      'createIndexedMatcher: this scorer has no indexed representation. ' +
        'Indexed search is available for dice.similarity and cosine.similarity.',
    )
  }
  // No runtime direction check: only a similarity metric declares the
  // capability, so a distance scorer is already refused above.
  const direction = compilation.direction
  const read = sequenceReader({ scorer, getText, normalize, missingItems }, false)
  // Flat and parallel rather than one row object per choice. Removing the
  // prepared representation promotes the bookkeeping: beside a Matcher's
  // profiles the per-choice rows were 0.3% of what it retained, and beside an
  // index they are the largest thing left.
  const kept: TItem[] = []
  // `null` while every key is its own position, which is an array with no gaps —
  // the common case, and the difference between a second array and nothing.
  let keys: unknown[] | null = null
  const builder = indexChoices()
  let position = 0
  const accept = (item: TItem, key: unknown): void => {
    const sequence = read(item)
    if (sequence === null) return
    if (keys === null && key !== position) {
      keys = []
      for (let index = 0; index < position; index++) keys.push(index)
    }
    if (keys !== null) keys.push(key)
    kept.push(item)
    builder.add(sequence)
    position++
  }
  if (Array.isArray(items)) {
    for (let key = 0; key < items.length; key++) accept(items[key], key)
  } else {
    for (const entry of collectionEntries(items)) accept(entry.item, entry.key)
  }
  const index: ChoiceIndex = builder.seal()
  const storedKeys = keys
  const keyOf = (id: number): unknown => (storedKeys === null ? id : storedKeys[id])
  const matchAt = (id: number, score: number): Match<TItem, unknown> => ({
    item: kept[id],
    key: keyOf(id),
    score,
  })

  // The same two decisions `createMatcher` makes before reaching a kernel: can
  // anything clear this threshold, and what does the kernel get told.
  //
  // Unconditionally the trusted forms, where `createMatcher` has to ask. Only a
  // built-in metric declares an index, and a built-in metric's bounds are its
  // own — the reason `createMatcher` asks is a custom scorer, which never
  // reaches here because it offers no capability to refuse.
  const impossible = (threshold: number | null): boolean =>
    impossibleTrustedThreshold(direction, compilation.bounds, threshold)
  const activeThreshold = (threshold: number | null): number | null =>
    trustedKernelThreshold(direction, compilation.bounds, threshold)

  /** What a query with no text to score is worth against every choice. */
  const missingScoreOf = (query: MaybeSequence, threshold: number | null): number =>
    compilation.score(query, '', threshold)

  const materialize = (found: SelectedChoices): Match<TItem, unknown>[] => {
    const matches: Match<TItem, unknown>[] = new Array(found.length)
    for (let at = 0; at < found.length; at++) {
      matches[at] = matchAt(found.ids[at], found.scores[at])
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
      const score = missingScoreOf(query, threshold)
      if (threshold !== null && score < threshold) return undefined
      return kept.length === 0 ? undefined : matchAt(0, score)
    }
    if (impossible(threshold)) return undefined
    const found = index.select(normalized, activeThreshold(threshold), 1)
    return found.length === 0 ? undefined : matchAt(found.ids[0], found.scores[0])
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
      if (threshold !== null && score < threshold) return []
      const length = limit === null ? kept.length : Math.min(kept.length, limit)
      const matches: Match<TItem, unknown>[] = new Array(length)
      for (let id = 0; id < length; id++) matches[id] = matchAt(id, score)
      return matches
    }
    if (impossible(threshold)) return []
    return materialize(index.select(normalized, activeThreshold(threshold), limit))
  }

  const searchIter = (
    query: MaybeSequence,
    call?: BestOptions,
  ): IterableIterator<Match<TItem, unknown>> => {
    // The threshold is read where the call is made and the query normalized
    // where iteration starts, both matching `createMatcher` — a caller who
    // mutates either between the two sees what they would see there.
    if (call !== undefined) assertOptionKeys(call, CALL_BEST_KEYS, 'matcher.searchIter')
    const threshold = optionalThreshold(call?.threshold)
    function* iterate(): Generator<Match<TItem, unknown>> {
      const normalized = normalizeQuery(query, normalize)
      if (normalized === null) {
        const score = missingScoreOf(query, threshold)
        if (threshold !== null && score < threshold) return
        for (let id = 0; id < kept.length; id++) yield matchAt(id, score)
        return
      }
      if (impossible(threshold)) return
      // The ids and scores are copied before the first yield, never streamed
      // from the index: the arrays a scan hands back are the index's own
      // scratch, so a `search` run between two `next()` calls would otherwise
      // rewrite what a live iterator is still walking.
      //
      // Two number arrays rather than the finished `Match` objects, which is
      // the same protection for less: a caller who stops after one result of
      // 100,000 measured 0.06x, and even a full drain 0.54x, against
      // materializing them all up front. Below ~100 results it costs 0.2µs
      // against a query that costs tens.
      const found = index.scan(normalized, activeThreshold(threshold))
      const ids = found.ids.slice(0, found.length)
      const scores = found.scores.slice(0, found.length)
      for (let at = 0; at < ids.length; at++) yield matchAt(ids[at], scores[at])
    }
    return iterate()
  }

  return Object.freeze({
    size: kept.length,
    scorer,
    best,
    search,
    searchIter,
  })
}
