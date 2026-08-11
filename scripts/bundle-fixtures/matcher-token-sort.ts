import { createMatcher, createScorer } from 'rapidfuzz-js'
import { tokenSortSimilarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(tokenSortSimilarity)
export const run = (query: string, choices: readonly string[]) =>
  createMatcher(choices, { scorer }).best(query)
