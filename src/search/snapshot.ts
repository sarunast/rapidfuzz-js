import { snapshotSequence, validateSequence } from '../core/sequence.js'
import type { MaybeSequence, Sequence } from '../core/types.js'
import type { MatcherOptions, Normalizer } from './types.js'

export function searchableSequence<T>(
  item: T,
  options: MatcherOptions<T, import('../core/types.js').Direction>,
  own: boolean,
): Sequence | null {
  const policy = options.missingItems ?? 'skip'
  if (item == null) {
    if (policy === 'skip') return null
    throw new TypeError('source item is missing')
  }
  const raw = options.getText === undefined ? item : options.getText(item)
  if (raw == null) {
    if (policy === 'skip') return null
    throw new TypeError('getText returned a missing value')
  }
  const sequence = validateSequence(raw)
  if (options.normalize === undefined) {
    return own ? snapshotSequence(sequence) : sequence
  }
  const normalized = options.normalize(sequence)
  if (normalized == null) throw new TypeError('normalize returned a missing value')
  const valid = validateSequence(normalized)
  return own ? snapshotSequence(valid) : valid
}

export function normalizeQuery(
  query: MaybeSequence,
  normalize: Normalizer | undefined,
): Sequence | null {
  if (query == null) return null
  const valid = validateSequence(query)
  if (normalize === undefined) return valid
  const normalized = normalize(valid)
  if (normalized == null) throw new TypeError('normalize returned a missing value')
  return validateSequence(normalized)
}

export function optionalThreshold(value: number | undefined): number | null {
  if (value === undefined) return null
  if (!Number.isFinite(value)) throw new RangeError('threshold must be finite')
  return value
}

export function resultLimit(value: number | null | undefined): number | null {
  if (value === null) return null
  const limit = value ?? 5
  if (!Number.isSafeInteger(limit) || limit < 0) {
    throw new RangeError('limit must be null or a non-negative integer')
  }
  return limit
}
