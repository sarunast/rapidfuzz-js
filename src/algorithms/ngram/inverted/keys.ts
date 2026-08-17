import { feasibleRadices, packGram, unpackGram } from '../key.js'

/**
 * Where a widening index writes the radix it was using and the one it needs, so
 * an extraction can report an overflow without allocating.
 */
export interface RadixWidening {
  from: number
  to: number | null
}

/**
 * Both extractors return a squared norm, which is never negative, or one of
 * these. An element the direct scheme cannot spell is `NEEDS_ORDINALS`; a
 * building index that has outgrown its radix is `NEEDS_WIDER_RADIX`, with the
 * rungs written into the `RadixWidening` it was given. A status rather than a
 * throw because both are ordinary states of a build, and because a query taking
 * the ordinal path takes it on every call.
 */
export const NEEDS_ORDINALS = -1
export const NEEDS_WIDER_RADIX = -2

export function radixFor(gramSize: number, element: number): number | null {
  if (element < 0) return null
  for (const radix of feasibleRadices(gramSize)) if (element < radix) return radix
  return null
}

export function repackKey(
  key: string | number,
  from: number,
  to: number | null,
  gramSize: number,
): string | number {
  if (typeof key === 'string') return key
  const elements: number[] = new Array<number>(gramSize)
  unpackGram(key, gramSize, from, elements)
  return to === null ? elements.join(',') : packGram(elements, 0, gramSize, to)
}

function joinGram(
  elements: ArrayLike<unknown>,
  start: number,
  gramSize: number,
): string | null {
  const first = elements[start]
  if (typeof first !== 'number' || !Number.isInteger(first)) return null
  let joined = String(first)
  for (let offset = 1; offset < gramSize; offset++) {
    const element = elements[start + offset]
    if (typeof element !== 'number' || !Number.isInteger(element)) return null
    joined += `,${element}`
  }
  return joined
}

function joinDigits(digits: ArrayLike<number>, start: number, gramSize: number): string {
  let joined = String(digits[start])
  for (let offset = 1; offset < gramSize; offset++) {
    joined += `,${digits[start + offset]}`
  }
  return joined
}

/**
 * The two halves of one spelling, for the one-time move to ordinal keys: a
 * packed key is positional base-`radix`, a joined one is its digits with commas
 * between them. Both exist so that the builder can re-key an index it has
 * already compacted without knowing which spelling it holds.
 */
export function decodeGramKey(
  key: string | number,
  gramSize: number,
  radix: number | null,
  output: number[],
): void {
  if (radix === null) {
    const digits = String(key).split(',')
    for (let index = 0; index < gramSize; index++) output[index] = Number(digits[index])
    return
  }
  unpackGram(Number(key), gramSize, radix, output)
}

export function encodeGramKey(
  digits: readonly number[],
  gramSize: number,
  radix: number | null,
): string | number {
  return radix === null
    ? joinDigits(digits, 0, gramSize)
    : packGram(digits, 0, gramSize, radix)
}

export function extractGrams(
  elements: ArrayLike<unknown>,
  gramSize: number,
  radix: number | null,
  widening: RadixWidening | null,
  keys: (string | number)[],
  counts: number[],
): number {
  keys.length = 0
  counts.length = 0
  const total = elements.length - gramSize + 1
  const seen = new Map<string | number, number>()
  let squaredNorm = 0
  for (let start = 0; start < total; start++) {
    let key: string | number
    if (radix === null) {
      const joined = joinGram(elements, start, gramSize)
      if (joined === null) return NEEDS_ORDINALS
      key = joined
    } else {
      let packed = 0
      let fits = true
      for (let offset = 0; offset < gramSize; offset++) {
        const value = elements[start + offset]
        if (typeof value !== 'number' || !Number.isInteger(value)) return NEEDS_ORDINALS
        if (value < 0 || value >= radix) {
          if (widening !== null) {
            widening.from = radix
            widening.to = radixFor(gramSize, value)
            return NEEDS_WIDER_RADIX
          }
          fits = false
          break
        }
        packed = packed * radix + value
      }
      if (fits) {
        key = packed
      } else {
        const joined = joinGram(elements, start, gramSize)
        if (joined === null) return NEEDS_ORDINALS
        key = joined
      }
    }
    const previous = seen.get(key)
    if (previous === undefined) {
      seen.set(key, counts.length)
      keys.push(key)
      counts.push(1)
      squaredNorm += 1
      continue
    }
    squaredNorm += 2 * counts[previous] + 1
    counts[previous]++
  }
  return squaredNorm
}

/**
 * The same key scheme over ordinals rather than elements, where `UNMATCHABLE`
 * poisons every window it falls in. A poisoned window counts toward the norm
 * and is keyed nowhere, which is what the exhaustive trie does with a gram
 * holding `NaN`. Tracking the last poisoned position tests each ordinal once
 * rather than once per window.
 */
export function extractOrdinalGrams(
  ordinals: readonly number[],
  gramSize: number,
  radix: number | null,
  widening: RadixWidening | null,
  keys: (string | number)[],
  counts: number[],
): number {
  keys.length = 0
  counts.length = 0
  const total = ordinals.length - gramSize + 1
  const seen = new Map<string | number, number>()
  const last = gramSize - 1
  let poisoned = -1
  for (let index = 0; index < last; index++) if (ordinals[index] < 0) poisoned = index
  let squaredNorm = 0
  for (let start = 0; start < total; start++) {
    const end = start + last
    if (ordinals[end] < 0) poisoned = end
    if (poisoned >= start) {
      squaredNorm += 1
      continue
    }
    let key: string | number
    if (radix === null) {
      key = joinDigits(ordinals, start, gramSize)
    } else {
      let packed = 0
      let fits = true
      for (let offset = 0; offset < gramSize; offset++) {
        const value = ordinals[start + offset]
        if (value >= radix) {
          if (widening !== null) {
            widening.from = radix
            widening.to = radixFor(gramSize, value)
            return NEEDS_WIDER_RADIX
          }
          fits = false
          break
        }
        packed = packed * radix + value
      }
      key = fits ? packed : joinDigits(ordinals, start, gramSize)
    }
    const previous = seen.get(key)
    if (previous === undefined) {
      seen.set(key, counts.length)
      keys.push(key)
      counts.push(1)
      squaredNorm += 1
      continue
    }
    squaredNorm += 2 * counts[previous] + 1
    counts[previous]++
  }
  return squaredNorm
}

/**
 * Rewrites query grams extracted over query-local ordinals into the direct keys
 * a direct index holds, dropping the grams no integer scheme can spell. A
 * dropped gram misses exactly as a key absent from the dictionary would, and
 * its contribution to the query's own norm was already counted.
 */
export function narrowToDirectKeys(
  elementsByOrdinal: readonly unknown[],
  localRadix: number | null,
  radix: number | null,
  gramSize: number,
  keys: (string | number)[],
  counts: number[],
): void {
  const ordinals = new Array<number>(gramSize)
  const digits = new Array<number>(gramSize)
  let write = 0
  for (let index = 0; index < keys.length; index++) {
    decodeGramKey(keys[index], gramSize, localRadix, ordinals)
    let fits = true
    for (let offset = 0; offset < gramSize; offset++) {
      const element = elementsByOrdinal[ordinals[offset]]
      if (typeof element !== 'number' || !Number.isInteger(element)) {
        fits = false
        break
      }
      if (radix !== null && (element < 0 || element >= radix)) {
        fits = false
        break
      }
      digits[offset] = element
    }
    if (!fits) continue
    keys[write] = encodeGramKey(digits, gramSize, radix)
    counts[write] = counts[index]
    write++
  }
  keys.length = write
  counts.length = write
}
