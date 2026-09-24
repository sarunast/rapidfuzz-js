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

// A kernel pruned at the exact complement `1 - cutoff` rejects a pair whose
// score equals the cutoff whenever `1 - cutoff` rounds past it. The slack only
// widens the pruning bound; the final check is still made against `cutoff`.
const COMPLEMENT_SLACK = 4 * Number.EPSILON

export function complementCutoff(cutoff: number | null | undefined): number {
  return cutoff == null ? 0 : Math.max(0, 1 - cutoff - COMPLEMENT_SLACK)
}

export function complementFraction(cutoff: number): number {
  return 1 - cutoff + COMPLEMENT_SLACK
}

export function distanceCutoffFor(
  kind: MetricScoreKind,
  rawCutoff: number | null | undefined,
  maximum: number,
  unbounded: number = Number.POSITIVE_INFINITY,
): number {
  if (rawCutoff == null) return unbounded
  switch (kind) {
    case 'distance':
      return rawDistanceBound(rawCutoff)
    case 'similarity':
      return maximum - Math.ceil(rawCutoff)
    case 'normalizedDistance':
      return rawCutoff * maximum
    case 'normalizedSimilarity':
      return complementFraction(rawCutoff) * maximum
  }
}

export function scoreFromDistance(
  kind: MetricScoreKind,
  distance: number,
  maximum: number,
  rawCutoff: number | null | undefined,
): number {
  switch (kind) {
    case 'distance':
      return distCutoff(distance, rawCutoff)
    case 'similarity':
      return simCutoff(maximum - distance, rawCutoff)
    case 'normalizedDistance':
      return normDistCutoff(normalizeDistance(distance, maximum), rawCutoff)
    case 'normalizedSimilarity':
      return normSimCutoff(1 - normalizeDistance(distance, maximum), rawCutoff)
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
