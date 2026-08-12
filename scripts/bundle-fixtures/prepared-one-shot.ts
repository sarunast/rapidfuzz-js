import { createScorer, searchIter, type PreparedChoiceOf } from 'rapidfuzz-js'
import { similarity } from 'rapidfuzz-js/fuzz'

const scorer = createScorer(similarity)

export interface Row {
  readonly prepared: PreparedChoiceOf<typeof scorer>
}

export const prepare = (choice: string): Row => ({
  prepared: scorer.prepareChoice(choice),
})

export const run = (query: string, rows: readonly Row[]) =>
  searchIter(query, rows, { scorer, getPrepared: (row) => row.prepared })
