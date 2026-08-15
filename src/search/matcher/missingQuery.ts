import { passesThreshold } from '../../core/scoring/threshold.js'
import type { Match } from '../results.js'
import type { ChoiceTable } from './choiceTable.js'
import { matchAt } from './choiceTable.js'

export function missingSimilarityBest<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
): Match<TItem, unknown> | undefined {
  if (!passesThreshold('similarity', score, threshold)) return undefined
  return table.items.length === 0 ? undefined : matchAt(table, 0, score)
}

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

export function* missingSimilarityMatches<TItem>(
  table: ChoiceTable<TItem>,
  score: number,
  threshold: number | null,
): Generator<Match<TItem, unknown>> {
  if (!passesThreshold('similarity', score, threshold)) return
  for (let id = 0; id < table.items.length; id++) yield matchAt(table, id, score)
}
