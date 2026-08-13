import type { PreparedKernel } from '../../../core/protocol.js'
import { pushHeap, replaceHeapRoot } from '../../shared/heap.js'
import { bestSimilarity } from './bestSimilarity.js'
import type { ScoredId } from './types.js'

function worse(left: ScoredId, right: ScoredId): boolean {
  return left.score < right.score || (left.score === right.score && left.id > right.id)
}

function result(entries: ScoredId[]): readonly ScoredId[] {
  entries.sort((a, b) => b.score - a.score || a.id - b.id)
  return entries
}

export function topSimilarity(
  prepared: readonly unknown[],
  score: PreparedKernel,
  threshold: number | null,
  limit: number | null,
  optimal: number | null,
): readonly ScoredId[] {
  if (limit === 0) return []
  if (limit === 1) {
    const found = bestSimilarity(prepared, score, threshold, optimal)
    return found === undefined ? [] : [found]
  }

  // A tie is broken by the id, which is the source position: nothing is skipped
  // before scoring, so the loop counter is the order a separate field used to
  // carry.
  if (limit !== null) {
    const heap: ScoredId[] = []
    let cutoff = threshold
    for (let id = 0; id < prepared.length; id++) {
      const value = score(prepared[id], cutoff)
      if (threshold !== null && value < threshold) continue
      if (heap.length < limit) {
        pushHeap(heap, { id, score: value }, worse)
        if (heap.length === limit) {
          cutoff = heap[0].score
          if (optimal !== null && cutoff === optimal) break
        }
        continue
      }
      // A tie loses on id, and every later candidate has a later id, so the
      // numeric test alone decides admission — no comparator call.
      if (value <= heap[0].score) continue
      replaceHeapRoot(heap, { id, score: value }, worse)
      cutoff = heap[0].score
      if (optimal !== null && cutoff === optimal) break
    }
    return result(heap)
  }

  const results: ScoredId[] = []
  for (let id = 0; id < prepared.length; id++) {
    const value = score(prepared[id], threshold)
    if (threshold === null || value >= threshold) {
      results.push({ id, score: value })
    }
  }
  return result(results)
}
