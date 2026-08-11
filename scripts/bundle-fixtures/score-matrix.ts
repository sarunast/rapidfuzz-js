import { createScorer, scoreMatrix } from 'rapidfuzz-js'
import { normalizedSimilarity } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(normalizedSimilarity)
export const run = (queries: readonly string[], choices: readonly string[]) =>
  scoreMatrix(queries, choices, { scorer })
