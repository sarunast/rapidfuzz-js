export type PreparedMetricKind = 'distance' | 'normalizedSimilarity'

function truncatedRawCutoff(cutoff: number): number {
  return Math.trunc(cutoff)
}

export function canonicalRawCutoff(cutoff: number | null | undefined): number | null {
  return cutoff == null ? null : truncatedRawCutoff(cutoff)
}

export function distanceCutoffFor(
  kind: PreparedMetricKind,
  rawCutoff: number | null | undefined,
  maximum: number,
): number {
  if (rawCutoff == null) return Number.POSITIVE_INFINITY
  return kind === 'distance' ? truncatedRawCutoff(rawCutoff) : (1 - rawCutoff) * maximum
}

export function distCutoff(distance: number, cutoff?: number | null): number {
  if (cutoff == null) return distance
  const bound = truncatedRawCutoff(cutoff)
  return distance <= bound ? distance : bound + 1
}

export function normSimCutoff(similarity: number, cutoff?: number | null): number {
  if (cutoff == null) return similarity
  return similarity >= cutoff ? similarity : 0
}

export function normalize(distance: number, maximum: number): number {
  return maximum === 0 ? 0 : distance / maximum
}
