export type PreparedMetricKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

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
  switch (kind) {
    case 'distance':
      return truncatedRawCutoff(rawCutoff)
    case 'similarity':
      return maximum - truncatedRawCutoff(rawCutoff)
    case 'normalizedDistance':
      return rawCutoff * maximum
    case 'normalizedSimilarity':
      return (1 - rawCutoff) * maximum
  }
}

export function distCutoff(distance: number, cutoff?: number | null): number {
  if (cutoff == null) return distance
  const bound = truncatedRawCutoff(cutoff)
  return distance <= bound ? distance : bound + 1
}

export function simCutoff(similarity: number, cutoff?: number | null): number {
  if (cutoff == null) return similarity
  const bound = truncatedRawCutoff(cutoff)
  return similarity >= bound ? similarity : 0
}

export function normDistCutoff(distance: number, cutoff?: number | null): number {
  if (cutoff == null) return distance
  return distance <= cutoff ? distance : 1
}

export function normSimCutoff(similarity: number, cutoff?: number | null): number {
  if (cutoff == null) return similarity
  return similarity >= cutoff ? similarity : 0
}

export function normalize(distance: number, maximum: number): number {
  return maximum === 0 ? 0 : distance / maximum
}
