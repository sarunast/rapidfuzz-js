import { snapshotSequence, validateSequence } from '../core/sequence.js'
import type { MaybeSequence, Sequence } from '../core/types.js'
import type { MatcherOptions, Normalizer } from './types.js'

export interface SequenceReader<T> {
  (item: T): Sequence | null
}

export function sequenceReader<T>(
  options: MatcherOptions<T, import('../core/types.js').Direction>,
  own: boolean,
): SequenceReader<T> {
  const policy = options.missingItems ?? 'skip'
  const retain = own ? snapshotSequence : (value: Sequence): Sequence => value
  const normalize = options.normalize

  if (options.getText === undefined) {
    if (normalize === undefined) {
      return (item) => {
        if (item == null) {
          if (policy === 'skip') return null
          throw new TypeError('source item is missing')
        }
        return retain(validateSequence(item))
      }
    }
    return (item) => {
      if (item == null) {
        if (policy === 'skip') return null
        throw new TypeError('source item is missing')
      }
      const normalized = normalize(validateSequence(item))
      if (normalized == null) throw new TypeError('normalize returned a missing value')
      return retain(validateSequence(normalized))
    }
  }

  const getText = options.getText
  if (normalize === undefined) {
    return (item) => {
      if (item == null) {
        if (policy === 'skip') return null
        throw new TypeError('source item is missing')
      }
      const raw = getText(item)
      if (raw == null) {
        if (policy === 'skip') return null
        throw new TypeError('getText returned a missing value')
      }
      return retain(validateSequence(raw))
    }
  }
  return (item) => {
    if (item == null) {
      if (policy === 'skip') return null
      throw new TypeError('source item is missing')
    }
    const raw = getText(item)
    if (raw == null) {
      if (policy === 'skip') return null
      throw new TypeError('getText returned a missing value')
    }
    const normalized = normalize(validateSequence(raw))
    if (normalized == null) throw new TypeError('normalize returned a missing value')
    return retain(validateSequence(normalized))
  }
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
