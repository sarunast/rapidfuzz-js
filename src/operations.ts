import type { MaybeSequence, Sequence } from './_common.js'
import { validateSequence } from './_metric.js'
import {
  scorerCompilation,
  type Direction,
  type Scorer,
  type ThresholdOptions,
} from './scorer.js'
import {
  allocateScores,
  buildScoreMatrix,
  roundHalfAwayFromZero,
  type ScoreArray,
  type ScoreArrayKind,
  type ScoreArrayOf,
  type ScoreMatrix,
} from './_scoreArray.js'

export interface Match<T, K> {
  readonly item: T
  readonly key: K
  readonly score: number
}

export type Items<T> =
  | readonly T[]
  | ReadonlyMap<unknown, T>
  | Iterable<T>
  | Readonly<Record<string, T>>

export type Normalizer = (value: Sequence) => MaybeSequence
export type MissingItemsPolicy = 'skip' | 'throw'

export interface MatcherOptions<T, D extends Direction> {
  readonly scorer: Scorer<D>
  readonly getText?: ((item: T) => MaybeSequence) | undefined
  readonly normalize?: Normalizer | undefined
  readonly missingItems?: MissingItemsPolicy | undefined
}

export interface BestOptions {
  readonly threshold?: number | undefined
}

export interface SearchOptions extends BestOptions {
  readonly limit?: number | null | undefined
}

export interface Matcher<T, K, D extends Direction = Direction> {
  readonly size: number
  readonly scorer: Scorer<D>
  best(query: MaybeSequence, options?: BestOptions): Match<T, K> | undefined
  search(query: MaybeSequence, options?: SearchOptions): readonly Match<T, K>[]
}

interface Entry<T, K> {
  readonly item: T
  readonly key: K
}

interface StoredEntry<T, K> extends Entry<T, K> {
  readonly sequence: Sequence
  readonly prepared: unknown
}

function assertItems(value: unknown): void {
  if (typeof value === 'string') {
    throw new TypeError('items must be a collection, not a single string')
  }
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('items must be an array, iterable, map, or object')
  }
}

function isMapLike(value: object): value is ReadonlyMap<unknown, unknown> {
  return (
    typeof Reflect.get(value, 'get') === 'function' &&
    typeof Reflect.get(value, 'has') === 'function' &&
    typeof Reflect.get(value, 'entries') === 'function' &&
    typeof Reflect.get(value, Symbol.iterator) === 'function'
  )
}

function isIterable(value: object): value is Iterable<unknown> {
  return typeof Reflect.get(value, Symbol.iterator) === 'function'
}

function* entriesOf<T>(items: Items<T>): Generator<Entry<T, unknown>> {
  assertItems(items)
  if (isMapLike(items)) {
    for (const [key, item] of items) yield { item, key }
    return
  }
  if (isIterable(items)) {
    let key = 0
    for (const item of items) {
      yield { item, key }
      key++
    }
    return
  }
  for (const [key, item] of Object.entries(items)) yield { item, key }
}

function snapshot(value: Sequence): Sequence {
  if (typeof value === 'string') return value
  const owned = new Array<unknown>(value.length)
  for (let i = 0; i < value.length; i++) owned[i] = value[i]
  return owned
}

function extracted<T>(item: T, getText: ((item: T) => MaybeSequence) | undefined): unknown {
  return getText === undefined ? item : getText(item)
}

function retainedSequence<T>(
  item: T,
  options: MatcherOptions<T, Direction>,
  own: boolean,
): Sequence | null {
  const policy = options.missingItems ?? 'skip'
  if (item == null) {
    if (policy === 'skip') return null
    throw new TypeError('source item is missing')
  }
  const raw = extracted(item, options.getText)
  if (raw == null) {
    if (policy === 'skip') return null
    throw new TypeError('getText returned a missing value')
  }
  const sequence = validateSequence(raw)
  if (options.normalize === undefined) return own ? snapshot(sequence) : sequence
  const normalized = options.normalize(sequence)
  if (normalized == null) throw new TypeError('normalize returned a missing value')
  const valid = validateSequence(normalized)
  return own ? snapshot(valid) : valid
}

function validateThreshold(value: number | undefined): number | null {
  if (value === undefined) return null
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

function validateLimit(value: number | null | undefined): number | null {
  if (value === null) return null
  const limit = value ?? 5
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new RangeError('limit must be null or a non-negative integer')
  }
  return limit
}

function qualifies(direction: Direction, score: number, threshold: number | null): boolean {
  if (threshold === null) return true
  return direction === 'similarity' ? score >= threshold : score <= threshold
}

function better(direction: Direction, candidate: number, current: number): boolean {
  return direction === 'similarity' ? candidate > current : candidate < current
}

function querySequence(query: MaybeSequence, normalize: Normalizer | undefined): Sequence | null {
  if (query == null) return null
  const valid = validateSequence(query)
  if (normalize === undefined) return valid
  const normalized = normalize(valid)
  if (normalized == null) throw new TypeError('normalize returned a missing value')
  return validateSequence(normalized)
}

function bestFromStored<T, K, D extends Direction>(
  query: MaybeSequence,
  stored: readonly StoredEntry<T, K>[],
  scorer: Scorer<D>,
  normalize: Normalizer | undefined,
  options: BestOptions | undefined,
): Match<T, K> | undefined {
  const threshold = validateThreshold(options?.threshold)
  const compilation = scorerCompilation(scorer)
  const normalized = querySequence(query, normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  let found: Match<T, K> | undefined
  for (const entry of stored) {
    const score =
      prepared === null
        ? compilation.score(query, entry.sequence, threshold)
        : prepared(entry.prepared, threshold)
    if (score === undefined || !qualifies(compilation.direction, score, threshold)) continue
    if (found === undefined || better(compilation.direction, score, found.score)) {
      found = { item: entry.item, key: entry.key, score }
    }
  }
  return found
}

function searchFromStored<T, K, D extends Direction>(
  query: MaybeSequence,
  stored: readonly StoredEntry<T, K>[],
  scorer: Scorer<D>,
  normalize: Normalizer | undefined,
  options: SearchOptions | undefined,
): readonly Match<T, K>[] {
  const threshold = validateThreshold(options?.threshold)
  const limit = validateLimit(options?.limit)
  if (limit === 0) return []
  const compilation = scorerCompilation(scorer)
  const normalized = querySequence(query, normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  const results: Array<Match<T, K> & { readonly order: number }> = []
  let order = 0
  for (const entry of stored) {
    const score =
      prepared === null
        ? compilation.score(query, entry.sequence, threshold)
        : prepared(entry.prepared, threshold)
    if (score !== undefined && qualifies(compilation.direction, score, threshold)) {
      results.push({ item: entry.item, key: entry.key, score, order })
    }
    order++
  }
  results.sort((a, b) => {
    const scoreOrder =
      compilation.direction === 'similarity' ? b.score - a.score : a.score - b.score
    return scoreOrder === 0 ? a.order - b.order : scoreOrder
  })
  const selected = limit === null ? results : results.slice(0, limit)
  return selected.map(({ item, key, score }) => ({ item, key, score }))
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
  const compilation = scorerCompilation(options.scorer)
  const stableOptions: MatcherOptions<T, Direction> = {
    scorer: options.scorer,
    ...(options.getText === undefined ? {} : { getText: options.getText }),
    ...(options.normalize === undefined ? {} : { normalize: options.normalize }),
    ...(options.missingItems === undefined
      ? {}
      : { missingItems: options.missingItems }),
  }
  const stored: StoredEntry<T, unknown>[] = []
  for (const entry of entriesOf(items)) {
    const sequence = retainedSequence(entry.item, stableOptions, true)
    if (sequence !== null) {
      stored.push({
        ...entry,
        sequence,
        prepared: compilation.prepareChoice(sequence),
      })
    }
  }
  const matcher: Matcher<T, unknown, D> = {
    size: stored.length,
    scorer: options.scorer,
    best: (query, callOptions) =>
      bestFromStored(query, stored, options.scorer, options.normalize, callOptions),
    search: (query, callOptions) =>
      searchFromStored(query, stored, options.scorer, options.normalize, callOptions),
  }
  return Object.freeze(matcher)
}

function streamingEntries<T, D extends Direction>(
  items: Items<T>,
  options: MatcherOptions<T, D>,
): Generator<Entry<T, unknown> & { readonly sequence: Sequence }> {
  const stableOptions: MatcherOptions<T, Direction> = {
    scorer: options.scorer,
    ...(options.getText === undefined ? {} : { getText: options.getText }),
    ...(options.normalize === undefined ? {} : { normalize: options.normalize }),
    ...(options.missingItems === undefined
      ? {}
      : { missingItems: options.missingItems }),
  }
  return (function* (): Generator<Entry<T, unknown> & { readonly sequence: Sequence }> {
    for (const entry of entriesOf(items)) {
      const sequence = retainedSequence(entry.item, stableOptions, false)
      if (sequence !== null) yield { ...entry, sequence }
    }
  })()
}

export function bestMatch<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & BestOptions,
): Match<T, unknown> | undefined {
  const threshold = validateThreshold(options.threshold)
  const compilation = scorerCompilation(options.scorer)
  const normalized = querySequence(query, options.normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  let found: Match<T, unknown> | undefined
  for (const entry of streamingEntries(items, options)) {
    const choice = compilation.prepareChoice(entry.sequence)
    const score =
      prepared === null
        ? compilation.score(query, entry.sequence, threshold)
        : prepared(choice, threshold)
    if (score === undefined || !qualifies(compilation.direction, score, threshold)) continue
    if (found === undefined || better(compilation.direction, score, found.score)) {
      found = { item: entry.item, key: entry.key, score }
    }
  }
  return found
}

export function search<T, D extends Direction>(
  query: MaybeSequence,
  items: Items<T>,
  options: MatcherOptions<T, D> & SearchOptions,
): readonly Match<T, unknown>[] {
  const limit = validateLimit(options.limit)
  if (limit === 0) return []
  const threshold = validateThreshold(options.threshold)
  const compilation = scorerCompilation(options.scorer)
  const normalized = querySequence(query, options.normalize)
  const prepared = normalized === null ? null : compilation.prepareQuery(normalized)
  const results: Array<Match<T, unknown> & { readonly order: number }> = []
  let order = 0
  for (const entry of streamingEntries(items, options)) {
    const choice = compilation.prepareChoice(entry.sequence)
    const score =
      prepared === null
        ? compilation.score(query, entry.sequence, threshold)
        : prepared(choice, threshold)
    if (score !== undefined && qualifies(compilation.direction, score, threshold)) {
      results.push({ item: entry.item, key: entry.key, score, order })
    }
    order++
  }
  results.sort((a, b) => {
    const scoreOrder =
      compilation.direction === 'similarity' ? b.score - a.score : a.score - b.score
    return scoreOrder === 0 ? a.order - b.order : scoreOrder
  })
  const selected = limit === null ? results : results.slice(0, limit)
  return selected.map(({ item, key, score }) => ({ item, key, score }))
}

export interface MatrixOptions<D extends Direction, K extends ScoreArrayKind = 'f64'> {
  readonly scorer: Scorer<D>
  readonly into?: K | undefined
}

function fillScores<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  scorer: Scorer<D>,
  store: ScoreArray,
  integral: boolean,
): void {
  const compilation = scorerCompilation(scorer)
  let offset = 0
  for (const query of queries) {
    const prepared = compilation.prepareQuery(validateSequence(query))
    for (const choice of choices) {
      const score = prepared(compilation.prepareChoice(validateSequence(choice)), null)
      store[offset++] = integral ? roundHalfAwayFromZero(score) : score
    }
  }
}

export function scoreMatrix<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, 'f64'>,
): ScoreMatrix<Float64Array>
export function scoreMatrix<D extends Direction, K extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, K> & { readonly into: K },
): ScoreMatrix<ScoreArrayOf[K]>
export function scoreMatrix<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, ScoreArrayKind>,
): ScoreMatrix<ScoreArray> {
  const kind = options.into ?? 'f64'
  return buildScoreMatrix(kind, queries.length, choices.length, 'scoreMatrix', (data, integral) =>
    fillScores(queries, choices, options.scorer, data, integral),
  )
}

export function scorePairs<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, 'f64'>,
): Float64Array
export function scorePairs<D extends Direction, K extends ScoreArrayKind>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, K> & { readonly into: K },
): ScoreArrayOf[K]
export function scorePairs<D extends Direction>(
  queries: readonly Sequence[],
  choices: readonly Sequence[],
  options: MatrixOptions<D, ScoreArrayKind>,
): ScoreArray {
  if (queries.length !== choices.length) {
    throw new RangeError('queries and choices must have the same length')
  }
  const kind = options.into ?? 'f64'
  const scores = allocateScores(kind, queries.length, 'scorePairs')
  const integral = kind !== 'f64' && kind !== 'f32'
  const compilation = scorerCompilation(options.scorer)
  for (let i = 0; i < queries.length; i++) {
    const query = validateSequence(queries[i])
    const choice = validateSequence(choices[i])
    const score = compilation.prepareQuery(query)(compilation.prepareChoice(choice), null)
    scores[i] = integral ? roundHalfAwayFromZero(score) : score
  }
  return scores
}

export type { ScoreArray, ScoreArrayKind, ScoreArrayOf, ScoreMatrix }
export type { ThresholdOptions }
