import type { Direction } from './types.js'

export function validateThreshold(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

export function qualifies(
  direction: Direction,
  score: number,
  threshold: number,
): boolean {
  return direction === 'similarity' ? score >= threshold : score <= threshold
}

export function impossibleTrustedThreshold(
  direction: Direction,
  bounds: readonly [number, number],
  threshold: number | null,
): boolean {
  return (
    threshold !== null &&
    (direction === 'similarity' ? threshold > bounds[1] : threshold < bounds[0])
  )
}

export function trustedKernelThreshold(
  direction: Direction,
  bounds: readonly [number, number],
  threshold: number | null,
): number | null {
  if (threshold === null) return null
  return direction === 'similarity'
    ? threshold <= bounds[0]
      ? null
      : threshold
    : threshold >= bounds[1]
      ? null
      : threshold
}
