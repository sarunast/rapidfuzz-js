import { feasibleRadices, packGram, unpackGram } from '../key.js'

export class OutOfRadix extends Error {
  constructor(
    readonly element: number,
    readonly radix: number,
  ) {
    super('gram element does not fit the packed key radix')
  }
}

function integerElement(element: unknown): number {
  if (typeof element !== 'number' || !Number.isInteger(element)) {
    throw new TypeError(
      `an indexed choice holds integer elements only, and one of them is ${String(element)}`,
    )
  }
  return element
}

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

function joinGram(elements: ArrayLike<unknown>, start: number, gramSize: number): string {
  let joined = String(integerElement(elements[start]))
  for (let offset = 1; offset < gramSize; offset++) {
    joined += `,${integerElement(elements[start + offset])}`
  }
  return joined
}

export function extractGrams(
  elements: ArrayLike<unknown>,
  gramSize: number,
  radix: number | null,
  widening: boolean,
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
      key = joinGram(elements, start, gramSize)
    } else {
      let packed = 0
      let fits = true
      for (let offset = 0; offset < gramSize; offset++) {
        const value = integerElement(elements[start + offset])
        if (value < 0 || value >= radix) {
          if (widening) throw new OutOfRadix(value, radix)
          fits = false
          break
        }
        packed = packed * radix + value
      }
      key = fits ? packed : joinGram(elements, start, gramSize)
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
