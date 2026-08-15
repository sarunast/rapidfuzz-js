import type { PreparedKernel } from '#core/scoring/compilation.js'

import type { ScoredId } from './types.js'

export function bestSimilarity(
  prepared: readonly unknown[],
  score: PreparedKernel,
  threshold: number | null,
  optimal: number | null,
): ScoredId | undefined {
  let bestId = -1
  let bestScore = Number.NEGATIVE_INFINITY
  let cutoff = threshold
  for (let id = 0; id < prepared.length; id++) {
    const value = score(prepared[id], cutoff)
    if (threshold !== null && value < threshold) continue
    if (bestId === -1 || value > bestScore) {
      bestId = id
      bestScore = value
      cutoff = value
      if (optimal !== null && value === optimal) break
    }
  }
  return bestId === -1 ? undefined : { id: bestId, score: bestScore }
}
