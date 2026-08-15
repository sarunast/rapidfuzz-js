import type { Sequence } from '../../core/types.js'

export { isSequence, validateSequence as asSequence } from '../../core/sequence.js'

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

export function isMissing(value: unknown): value is null | undefined {
  return value == null
}

export function elementsEqual(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  if (a.length !== b.length) return false
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false
  }
  return true
}
