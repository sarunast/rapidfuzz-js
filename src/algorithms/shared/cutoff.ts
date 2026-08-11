export type MetricScoreKind =
  | 'distance'
  | 'similarity'
  | 'normalizedDistance'
  | 'normalizedSimilarity'

/**
 * Integer distance bound. Truncation equals flooring only for `cutoff >= 0`;
 * impossible negative thresholds are settled by the batch and search
 * boundaries before any kernel runs — see `rejectedScore` and
 * `impossibleTrustedThreshold`.
 */
function rawDistanceBound(cutoff: number): number {
  return Math.trunc(cutoff)
}

export function canonicalRawCutoff(cutoff: number | null | undefined): number | null {
  return cutoff == null ? null : rawDistanceBound(cutoff)
}

/**
 * The integral similarity a fractional cutoff really demands.
 *
 * Raw similarities are counts, so `similarity >= cutoff` is exactly
 * `similarity >= ceil(cutoff)`. Truncating instead admitted a similarity of 2
 * against a cutoff of 2.5 — and, because every similarity kernel derives its
 * distance budget from this, let the kernel search one edit further than the
 * caller asked for. The public threshold is any finite number, so the fraction
 * has to be honoured rather than rounded away.
 */
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
      // `similarity >= cutoff` is `distance <= maximum - cutoff`, and both are
      // counts, so the budget is the floor of that difference.
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
  // Compared against the cutoff as given. Truncating here was the second half
  // of the fractional-cutoff bug: a caller asking for 2.5 got 2 accepted.
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
