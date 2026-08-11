import type { Direction, MaybeSequence, MissingPolicy, Sequence } from './types.js'

/**
 * The longest array JavaScript can hold. Without it `{ length: 2 ** 53 - 1 }`
 * passes here and fails in {@link snapshotSequence} instead, as a `RangeError`
 * about our own array. A representability bound, not a resource one.
 */
const MAX_SEQUENCE_LENGTH = 0xffff_ffff

/**
 * A callable is not a sequence, though every function has a `length`: accepting
 * one would score a misplaced argument instead of reporting it.
 */
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
