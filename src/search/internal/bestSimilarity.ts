import type { PreparedKernel } from '../../core/protocol.js'
import type { ScoredId } from './types.js'

export function bestSimilarity(
  prepared: readonly unknown[],
  score: PreparedKernel,
  threshold: number | null,
  optimal: number | null,
): ScoredId | undefined {
  // The winner is carried as an id, so a run of improvements allocates one
  // result rather than one per improvement.
  //
  // What makes the first candidate a winner is `bestId`, not the sentinel: a
  // custom metric may return the sentinel as a real score, and a bare
  // `value > bestScore` would then select nothing. The sentinel is only here
  // to mirror `bestDistance`.
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
