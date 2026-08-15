export type MetricScoreKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

function rawDistanceBound(cutoff: number): number {
  return Math.trunc(cutoff)
}

export function canonicalRawCutoff(cutoff: number | null | undefined): number | null {
  return cutoff == null ? null : rawDistanceBound(cutoff)
}

export function canonicalSimilarityCutoff(
  cutoff: number | null | undefined,
): number | null {
  return cutoff == null ? null : Math.ceil(cutoff)
}

export function distanceCutoffFor(
  kind: MetricScoreKind,
  rawCutoff: number | null | undefined,
  maximum: number,
): number {
  if (rawCutoff == null) return Number.POSITIVE_INFINITY
  switch (kind) {
    case 'distance':
      return rawDistanceBound(rawCutoff)
    case 'similarity':
      return maximum - Math.ceil(rawCutoff)
    case 'normalizedDistance':
      return rawCutoff * maximum
    case 'normalizedSimilarity':
      return (1 - rawCutoff) * maximum
  }
}

export function distCutoff(distance: number, cutoff?: number | null): number {
  if (cutoff == null) return distance
  const bound = rawDistanceBound(cutoff)
  return distance <= bound ? distance : bound + 1
}

export function simCutoff(similarity: number, cutoff?: number | null): number {
  if (cutoff == null) return similarity
  return similarity >= cutoff ? similarity : 0
}

export function normDistCutoff(distance: number, cutoff?: number | null): number {
  if (cutoff == null) return distance
  return distance <= cutoff ? distance : 1
}

export function normSimCutoff(similarity: number, cutoff?: number | null): number {
  if (cutoff == null) return similarity
  return similarity >= cutoff ? similarity : 0
}

export function normalizeDistance(distance: number, maximum: number): number {
  return maximum === 0 ? 0 : distance / maximum
}
