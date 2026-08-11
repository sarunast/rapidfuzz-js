export type Sequence = string | ArrayLike<unknown>
export type MaybeSequence = Sequence | null | undefined
export type Direction = 'similarity' | 'distance'
export type MissingPolicy = 'compatible' | 'throw'
export type Normalizer = (value: Sequence) => MaybeSequence

export interface SimilarityConfiguration {
  readonly missing?: MissingPolicy | undefined
}
