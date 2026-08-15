import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Normalizer,
  Sequence,
} from './types.js'

const MAX_SEQUENCE_LENGTH = 0xffff_ffff

export function isSequence(value: unknown): value is Sequence {
  if (typeof value === 'string') return true
  if (typeof value !== 'object' || value === null || !('length' in value)) return false
  const length = value.length
  return (
    typeof length === 'number' &&
    Number.isSafeInteger(length) &&
    length >= 0 &&
    length <= MAX_SEQUENCE_LENGTH
  )
}

export function validateSequence(value: unknown): Sequence {
  if (!isSequence(value)) {
    throw new TypeError('expected a string or an array-like sequence')
  }
  return value
}

export function normalizeSequence(value: Sequence, normalize: Normalizer): Sequence {
  const normalized = normalize(value)
  if (normalized == null) throw new TypeError('normalize returned a missing value')
  return validateSequence(normalized)
}

export function validatePair(
  a: MaybeSequence,
  b: MaybeSequence,
  direction: Direction,
  missing: MissingPolicy,
): readonly [Sequence, Sequence] | null {
  if (a == null || b == null) {
    if (direction === 'similarity' && missing === 'compatible') return null
    throw new TypeError('missing sequences are not supported by this scorer')
  }
  return [validateSequence(a), validateSequence(b)]
}

export function snapshotSequence(value: Sequence): Sequence {
  if (typeof value === 'string') return value
  const length = value.length
  const owned = new Array<unknown>(length)
  for (let i = 0; i < length; i++) owned[i] = value[i]
  return owned
}
