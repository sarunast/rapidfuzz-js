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
