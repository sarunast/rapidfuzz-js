import type { Sequence } from '../types.js'
import type { ChoiceIndexBuilder } from './choiceIndex.js'

/** Borrowed candidate ids, valid until the next query on the same index. */
export interface CandidateChoices {
  readonly ids: Uint32Array
  readonly length: number
}

/**
 * A similarity-only shortlist. False positives are allowed; false negatives are not:
 * every choice whose true score is at least `threshold` must be returned.
 */
export interface CandidateIndex {
  /** Returned ids are ascending and borrowed until the next call. */
  candidates(query: Sequence, threshold: number): CandidateChoices
}

/**
 * One-shot builder. `add` consumes the sequence state at that instant; mutation by
 * the caller afterwards cannot affect the sealed index.
 */
export interface CandidateIndexBuilder {
  add(choice: Sequence): void
  seal(): CandidateIndex
}

/** Adapt an exact similarity index by discarding the scores returned by `scan`. */
export function candidateBuilderFromExactIndex(
  exact: () => ChoiceIndexBuilder,
): () => CandidateIndexBuilder {
  return () => {
    const builder = exact()
    let sealed = false
    return {
      add(choice) {
        if (sealed) throw new TypeError('candidate index builder is already sealed')
        builder.add(choice)
      },
      seal() {
        if (sealed) throw new TypeError('candidate index builder is already sealed')
        sealed = true
        const index = builder.seal()
        return {
          candidates(query, threshold) {
            const found = index.scan(query, threshold)
            return { ids: found.ids, length: found.length }
          },
        }
      },
    }
  }
}
