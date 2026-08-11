import type { Sequence } from '../../core/types.js'
import type { Processor } from './types.js'

function toCodePoints(value: string): Uint32Array {
  const length = value.length
  const output = new Uint32Array(length)
  let size = 0
  for (let index = 0; index < length; index++) {
    const high = value.charCodeAt(index)
    if (high >= 0xd800 && high <= 0xdbff && index + 1 < length) {
      const low = value.charCodeAt(index + 1)
      if (low >= 0xdc00 && low <= 0xdfff) {
        output[size++] = (high - 0xd800) * 0x400 + (low - 0xdc00) + 0x10000
        index++
        continue
      }
    }
    output[size++] = high
  }
  return size === length ? output : output.subarray(0, size)
}

// This regexp is load-bearing: on Latin-1 strings V8 can reject it without a
// character-by-character JS scan, and it measured ~22x faster than that loop.
const SURROGATE_PAIR = /[\uD800-\uDBFF][\uDC00-\uDFFF]/

export function hasSurrogatePair(value: string): boolean {
  return SURROGATE_PAIR.test(value)
}

function convElement(value: unknown): unknown {
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

export function isSequence(value: unknown): value is Sequence {
  if (typeof value === 'string') return true
  if (typeof value !== 'object' || value === null || !('length' in value)) return false
  const length = value.length
  return typeof length === 'number' && Number.isSafeInteger(length) && length >= 0
}

function convPair(
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

export function isMissing(value: unknown): value is null | undefined {
  return value == null
}

export function asSequence(value: unknown): Sequence {
  if (!isSequence(value))
    throw new TypeError('expected a string or an array-like sequence')
  return value
}

export function conv(
  left: Sequence,
  right: Sequence,
  processor?: Processor,
): [ArrayLike<unknown>, ArrayLike<unknown>] {
  if (processor == null) return convPair(left, right)
  const processedLeft = processor(left)
  const processedRight = processor(right)
  if (!isSequence(processedLeft) || !isSequence(processedRight)) {
    throw new TypeError('processor must return a string or an array-like sequence')
  }
  return convPair(processedLeft, processedRight)
}
