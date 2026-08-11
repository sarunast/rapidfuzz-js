import { createScorer } from 'rapidfuzz-js'
import { normalizedSimilarity } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(normalizedSimilarity)
export const run = (a: string, b: string): number => scorer.score(a, b)
