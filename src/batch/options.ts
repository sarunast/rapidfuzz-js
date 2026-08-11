import { validateThreshold } from '../core/threshold.js'

export interface ResolvedBatchOptions {
  readonly threshold: number | null
  readonly multiplier: number
}

export function resolveBatchOptions(
  threshold: number | undefined,
  multiplier: number | undefined,
): ResolvedBatchOptions {
  const resolvedMultiplier = multiplier ?? 1
  if (!Number.isFinite(resolvedMultiplier)) {
    throw new RangeError('scoreMultiplier must be finite')
  }
  return {
    threshold: threshold === undefined ? null : validateThreshold(threshold),
    multiplier: resolvedMultiplier,
  }
}
