export type Sequence = string | ArrayLike<unknown>
export type MaybeSequence = Sequence | null | undefined
export type Direction = 'similarity' | 'distance'
export type MissingPolicy = 'compatible' | 'throw'

export interface SimilarityConfiguration {
  readonly missing?: MissingPolicy | undefined
}
