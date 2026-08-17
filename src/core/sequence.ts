import type {
  Direction,
  MaybeSequence,
  MissingPolicy,
  Normalizer,
  Sequence,
} from './types.js'

export const MAX_SEQUENCE_LENGTH = 0xffff_ffff

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

function toCodePoints(value: string): Uint16Array | Uint32Array {
  const length = value.length
  let output: Uint16Array | Uint32Array = new Uint16Array(length)
  let size = 0
  for (let index = 0; index < length; index++) {
    const high = value.charCodeAt(index)
    if (high >= 0xd800 && high <= 0xdbff && index + 1 < length) {
      const low = value.charCodeAt(index + 1)
      if (low >= 0xdc00 && low <= 0xdfff) {
        if (output instanceof Uint16Array) {
          const promoted = new Uint32Array(length)
          promoted.set(output.subarray(0, size))
          output = promoted
        }
        output[size++] = (high - 0xd800) * 0x400 + (low - 0xdc00) + 0x10000
        index++
        continue
      }
    }
    output[size++] = high
  }
  return size === length ? output : output.subarray(0, size)
}

const SURROGATE_PAIR = /[\uD800-\uDBFF][\uDC00-\uDFFF]/

export function hasSurrogatePair(value: string): boolean {
  return SURROGATE_PAIR.test(value)
}

export function convElement(value: unknown): unknown {
  if (typeof value !== 'string') return value
  if (value.length === 1) return value.charCodeAt(0)
  if (value.length === 2) {
    const high = value.charCodeAt(0)
    const low = value.charCodeAt(1)
    if (high >= 0xd800 && high <= 0xdbff && low >= 0xdc00 && low <= 0xdfff) {
      return (high - 0xd800) * 0x400 + (low - 0xdc00) + 0x10000
    }
  }
  return value
}

export function convSequence(value: Sequence): ArrayLike<unknown> {
  if (typeof value === 'string') return toCodePoints(value)
  if (ArrayBuffer.isView(value)) return value
  const length = value.length
  const output = new Array<unknown>(length)
  for (let index = 0; index < length; index++) output[index] = convElement(value[index])
  return output
}

export function convPair(
  left: Sequence,
  right: Sequence,
): [ArrayLike<unknown>, ArrayLike<unknown>] {
  if (typeof left === 'string' && typeof right === 'string') {
    if (!hasSurrogatePair(left) && !hasSurrogatePair(right)) return [left, right]
    return [toCodePoints(left), toCodePoints(right)]
  }
  return [convSequence(left), convSequence(right)]
}

export function scorerSequence(value: Sequence): ArrayLike<unknown> {
  return typeof value === 'string' && !hasSurrogatePair(value)
    ? value
    : convSequence(value)
}

export function alignRepresentation(
  value: ArrayLike<unknown>,
  other: ArrayLike<unknown>,
): ArrayLike<unknown> {
  return typeof value === 'string' && typeof other !== 'string'
    ? convSequence(value)
    : value
}

export function queryAligner(
  query: ArrayLike<unknown>,
): (choice: ArrayLike<unknown>) => ArrayLike<unknown> {
  let converted: ArrayLike<unknown> | null = null
  return (choice) =>
    typeof query === 'string' && typeof choice !== 'string'
      ? (converted ??= convSequence(query))
      : query
}

export function isMissing(value: unknown): value is null | undefined {
  return value == null
}

export function isUnmatchableElement(value: unknown): boolean {
  return typeof value === 'number' && Number.isNaN(value)
}

export function elementsEqual(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  if (a.length !== b.length) return false
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false
  }
  return true
}

export function maxSequenceLength(a: ArrayLike<unknown>, b: ArrayLike<unknown>): number {
  return Math.max(a.length, b.length)
}
