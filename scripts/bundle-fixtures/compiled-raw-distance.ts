import { createScorer } from 'rapidfuzz-js'
import { distance } from 'rapidfuzz-js/levenshtein'

const scorer = createScorer(distance)
export const run = (a: string, b: string): number => scorer.score(a, b)
