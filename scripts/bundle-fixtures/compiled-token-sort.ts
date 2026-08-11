import { createScorer } from 'rapidfuzz-js'
import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortSimilarity)
export const run = (a: string, b: string): number => scorer.score(a, b)
