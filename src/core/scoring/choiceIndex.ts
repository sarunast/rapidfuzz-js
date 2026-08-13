import type { Sequence } from '../types.js'

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
   * @throws {TypeError} If the choice holds an element that is not an integer,
   * or if the builder is already sealed.
   * @throws {RangeError} If the choice takes the index past what its ids can
   * address — 4,294,967,295 choices, posting entries, or grams in one choice.
   */
  add(choice: Sequence): void
  /** @throws {TypeError} If called twice. */
  seal(): ChoiceIndex
}
