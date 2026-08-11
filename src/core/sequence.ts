import type { Direction, MaybeSequence, MissingPolicy, Sequence } from './types.js'

export function isSequence(value: unknown): value is Sequence {
  if (typeof value === 'string') return true
  if (typeof value !== 'object' || value === null || !('length' in value)) return false
  const length = value.length
  return typeof length === 'number' && Number.isSafeInteger(length) && length >= 0
}

export function validateSequence(value: unknown): Sequence {
  if (!isSequence(value)) {
    throw new TypeError('expected a string or an array-like sequence')
  }
  return value
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
  const owned = new Array<unknown>(value.length)
  for (let i = 0; i < value.length; i++) owned[i] = value[i]
  return owned
}
