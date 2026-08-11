import { createScorer, searchIter } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(similarity)
export const run = (query: string, choices: readonly string[]) =>
  searchIter(query, choices, { scorer })
