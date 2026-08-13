import { passesThreshold } from '../../core/scoring/threshold.js'
import type { Match } from '../results.js'
import type { ChoiceTable } from './choiceTable.js'
import { matchAt } from './choiceTable.js'

// What every search returns for a query with no text to score: one score
// against every choice, so the only questions left are whether it clears the
// threshold and how many results the caller asked for. Both Matchers answer
// them here rather than each writing the three shapes out.
//
// Only a similarity scorer arrives: a distance metric refuses the pair in
// `validatePair` before a score exists, which is why the direction is written
// rather than read from the compilation.

/** The first choice, since one score cannot rank them. */
export function missingSimilarityBest<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
): Match<TItem, unknown> | undefined {
  if (!passesThreshold('similarity', score, threshold)) return undefined
  return table.items.length === 0 ? undefined : matchAt(table, 0, score)
}

/** Every choice in collection order, cut to `limit` where there is one. */
export function missingSimilarityTop<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
  limit: number | null,
): readonly Match<TItem, unknown>[] {
  if (!passesThreshold('similarity', score, threshold)) return []
  const count = table.items.length
  const length = limit === null ? count : Math.min(count, limit)
  const matches: Match<TItem, unknown>[] = new Array(length)
  for (let id = 0; id < length; id++) matches[id] = matchAt(table, id, score)
  return matches
}

/** {@link missingSimilarityTop} unbounded, built one `Match` at a time. */
export function* missingSimilarityMatches<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
): Generator<Match<TItem, unknown>> {
  if (!passesThreshold('similarity', score, threshold)) return
  for (let id = 0; id < table.items.length; id++) yield matchAt(table, id, score)
}
