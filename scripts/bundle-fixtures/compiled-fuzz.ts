import { createScorer } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(similarity)
export const run = (a: string, b: string): number => scorer.score(a, b)
