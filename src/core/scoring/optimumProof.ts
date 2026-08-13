import type { Sequence } from '../types.js'

/**
 * A shortcut to the choices that score a metric's optimum, without scoring any
 * of the others.
 *
 * Some metrics can name their perfect matches structurally. Token-set
 * similarity is the example: it answers 100 exactly when one non-empty token set
 * contains the other, so an index over tokens finds every perfect match without
 * comparing a single pair. A scan already stops at the first optimal score —
 * `bestSimilarity` breaks there — so what this saves is the candidates *before*
 * that one, which is most of a large collection.
 *
 * **Exact, or absent.** Every answer must be what scoring the whole collection
 * would have produced, including the tie order: ids ascend, because a
 * collection-order tie goes to the earliest. A proof that cannot be certain
 * says so rather than guessing.
 */
export interface OptimumProof {
  /**
   * The earliest id scoring the metric's optimum, or `undefined` when this
   * query is not settled.
   *
   * `undefined` covers both "no choice reaches the optimum" and "I decline to
   * answer": the caller does the same thing either way — scan — so the two are
   * one value rather than a distinction nothing can act on.
   */
  best(query: Sequence): number | undefined

  /**
   * Exactly `limit` ids at the optimum, ascending, when that settles a top-k
   * search outright — otherwise `undefined`.
   *
   * Fewer than `limit` is `undefined` rather than a short array: a partial
   * answer still needs the collection scored to fill the rest, so it settles
   * nothing. Requiring the full count here keeps that invariant in one place
   * instead of at every call site.
   */
  top(query: Sequence, limit: number): readonly number[] | undefined
}
