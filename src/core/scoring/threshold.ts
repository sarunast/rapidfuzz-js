import type { Direction } from '../types.js'
import type { MetricCompilation } from './compilation.js'

export function validateThreshold(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

export function optionalThreshold(value: number | undefined): number | null {
  return value === undefined ? null : validateThreshold(value)
}

export function qualifies(
  direction: Direction,
  score: number,
  threshold: number,
): boolean {
  return direction === 'similarity' ? score >= threshold : score <= threshold
}

export function passesThreshold(
  direction: Direction,
  score: number,
  threshold: number | null,
): boolean {
  return (
    threshold === null ||
    (direction === 'similarity' ? score >= threshold : score <= threshold)
  )
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

export function impossibleThreshold(
  compilation: MetricCompilation<Direction>,
  threshold: number | null,
): boolean {
  return (
    compilation.trusted &&
    impossibleTrustedThreshold(compilation.direction, compilation.bounds, threshold)
  )
}

export function kernelThreshold(
  compilation: MetricCompilation<Direction>,
  threshold: number | null,
): number | null {
  return compilation.trusted
    ? trustedKernelThreshold(compilation.direction, compilation.bounds, threshold)
    : threshold
}

export function knownOptimum(compilation: MetricCompilation<Direction>): number | null {
  if (!compilation.trusted) return null
  return compilation.direction === 'similarity'
    ? compilation.bounds[1]
    : compilation.bounds[0]
}
