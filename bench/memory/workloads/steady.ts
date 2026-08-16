import { ordinaryQuery, type IndexedMatcherWorkload, unrelatedQuery } from './shared.ts'

export const STEADY_BATCH_SIZE = 5_000

export interface SteadyTally {
  best: number
  limit5: number
  limit100: number
  unrelated: number
  unlimited: number
}

export function emptySteadyTally(): SteadyTally {
  return { best: 0, limit5: 0, limit100: 0, unrelated: 0, unlimited: 0 }
}

/** One named phase that preserves the committed operation mix at any batch size. */
export function runSteadyBatch(
  matcher: IndexedMatcherWorkload,
  batch: number,
  operations = STEADY_BATCH_SIZE,
  tally: SteadyTally = emptySteadyTally(),
): SteadyTally {
  const first = batch * operations
  for (let offset = 0; offset < operations; offset++) {
    const operation = first + offset
    // Reserve the last operation for the public unlimited search. Scale the
    // other 4,999 canonical slots across the remainder of a custom batch.
    const slot =
      operations === 1 ? 4_999 : Math.floor((offset * 4_999) / (operations - 1))
    if (slot < 2_105) {
      matcher.best(ordinaryQuery(operation))
      tally.best++
    } else if (slot < 3_684) {
      matcher.search(ordinaryQuery(operation), { limit: 5 })
      tally.limit5++
    } else if (slot < 4_473) {
      matcher.search(ordinaryQuery(operation), { limit: 100 })
      tally.limit100++
    } else if (slot < 4_999) {
      matcher.best(unrelatedQuery(operation))
      tally.unrelated++
    } else {
      matcher.search(ordinaryQuery(operation), { limit: null })
      tally.unlimited++
    }
  }
  return tally
}
