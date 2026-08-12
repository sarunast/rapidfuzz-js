export type Sequence = string | ArrayLike<unknown>
export type MaybeSequence = Sequence | null | undefined
export type Direction = 'similarity' | 'distance'
/**
 * What a similarity scorer does with a missing operand.
 *
 * `'compatible'` scores the pair `0`, including `null` against `null`: two
 * unknowns are not evidence of a match, and a perfect score there would put
 * every missing record at the top of a search or merge them in a dedup.
 * `'throw'` is for callers who would rather hear that the value never arrived.
 * Distance scorers have no such choice — there is no distance to report.
 */
export type MissingPolicy = 'compatible' | 'throw'
export type Normalizer = (value: Sequence) => MaybeSequence

export interface SimilarityConfiguration {
  readonly missing?: MissingPolicy | undefined
}
