import type { AnyBrand } from './prepared.js'
import type { Direction, MaybeSequence, Sequence } from './types.js'

export const COMPILE: unique symbol = Symbol('rapidfuzz.metric.compile')

export interface PreparedKernel {
  (choice: unknown, threshold: number | null): number
}

/**
 * Ids the index chose, and the scores it computed for them.
 *
 * **Borrowed**: the arrays are the index's own scratch and stay valid only until
 * the next call on the same index, the way `prepareChoice` borrows its sequence.
 * A caller that outlives one call — a generator, say — has to copy first.
 *
 * `length` is the count. Both arrays may be longer, and everything past `length`
 * is whatever the previous query left there.
 */
export interface SelectedChoices {
  /**
   * Always `Uint32Array`, whatever the index stores internally: how wide a
   * posting id is depends on the corpus size and is nobody else's business, and
   * a result's representation should not change because a collection grew.
   */
  readonly ids: Uint32Array
  /** Aligned with {@link ids}: `scores[n]` is the score of `ids[n]`. */
  readonly scores: Float64Array
  /** How many entries of {@link ids} and {@link scores} this call filled. */
  readonly length: number
}

/**
 * A sealed corpus-wide representation, ready to answer queries and impossible to
 * add to.
 *
 * Both operations are **exact**: they return what scoring every choice
 * one-by-one would have returned, to the bit — including the choices that share
 * nothing with the query, which score `0` and still come back whenever the
 * threshold admits them. An index that answered only what it touched would drop
 * rows the exhaustive path returns.
 */
export interface ChoiceIndex {
  /**
   * The best `limit` choices, ordered by descending score and then by ascending
   * id — the order a threshold-and-limit search returns. `null` means no limit.
   */
  select(query: Sequence, threshold: number | null, limit: number | null): SelectedChoices
  /**
   * Every qualifying choice in ascending id, which is collection order.
   *
   * Separate from {@link select} rather than a re-sort of it: a streaming search
   * promises collection order, and ordering by score and sorting back would
   * both cost more and, where untouched choices qualify at `0`, put them after
   * the rest instead of interleaved among them where their ids belong.
   */
  scan(query: Sequence, threshold: number | null): SelectedChoices
}

/**
 * Builds a {@link ChoiceIndex} one choice at a time.
 *
 * **One-shot**: every `add`, then one `seal`. Adding after sealing, or sealing
 * twice, is a `TypeError` — the shape worth relying on is a mutable builder that
 * becomes an immutable index exactly once.
 */
export interface ChoiceIndexBuilder {
  /**
   * Ids are the call order: the first choice added is `0`, the next `1`. The
   * caller decides what to add and therefore what each id means, so a collection
   * with gaps numbers only what it kept.
   *
   * @throws {TypeError} If the choice cannot be indexed — an element that is not
   * an integer, or a size past what the representation can address.
   */
  add(choice: Sequence): void
  /** @throws {TypeError} If called twice. */
  seal(): ChoiceIndex
}

interface Compilation<TDirection extends Direction, TBrand = AnyBrand> {
  readonly direction: TDirection
  readonly bounds: readonly [number, number]
  readonly symmetric: boolean
  readonly score: (a: MaybeSequence, b: MaybeSequence, threshold: number | null) => number
  readonly rawScore: (a: Sequence, b: Sequence, threshold: number | null) => number
  readonly prepareQuery: (query: Sequence) => PreparedKernel
  // Borrows the sequence: the result lives no longer than the loop that made
  // it, so preparation may keep a reference into the caller's data.
  readonly prepareChoice: (choice: Sequence) => unknown
  // Owns the sequence, for a handle that outlives the call. Whether that costs
  // a copy is the preparation's to know — most already copy what they convert.
  readonly prepareOwnedChoice: (choice: Sequence) => unknown
  // Identity a prepared choice is checked against; shared by compilations
  // whose preparation is compatible, fresh when the recorded configuration
  // differs or for a custom scorer. `missing` keeps the shared one — it
  // decides which pairs are refused and never reaches preparation.
  readonly preparedChoiceKey: object
  // Builds one corpus-wide representation for a whole collection instead of a
  // handle per choice. Absent on every metric that has no such representation,
  // which is what an indexed search refuses on. It takes no choice count: the
  // caller cannot know how many choices it will keep until it has read them
  // all, so ids come from the order they arrive in.
  readonly indexChoices?: (() => ChoiceIndexBuilder) | undefined
  // Never assigned and never read: it exists so a scorer's handles carry the
  // metric that made them into the type system.
  readonly preparedChoiceBrand?: TBrand
}

export interface TrustedMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
> extends Compilation<TDirection, TBrand> {
  readonly trusted: true
  readonly validate: (a: MaybeSequence, b: MaybeSequence) => void
}

export interface CustomMetricCompilation<
  TDirection extends Direction,
  TBrand = AnyBrand,
> extends Compilation<TDirection, TBrand> {
  readonly trusted: false
}

export type MetricCompilation<TDirection extends Direction, TBrand = AnyBrand> =
  | TrustedMetricCompilation<TDirection, TBrand>
  | CustomMetricCompilation<TDirection, TBrand>
