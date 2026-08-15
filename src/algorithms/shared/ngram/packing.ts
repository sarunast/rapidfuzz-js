import { packGram } from './key.js'

export type ElementDomain = 'number' | 'char'

export function packingDigit(
  element: unknown,
  domain: ElementDomain,
  radix: number,
): number {
  if (domain === 'number') {
    if (typeof element !== 'number' || !Number.isInteger(element)) return -1
    return element < 0 || element >= radix ? -1 : element
  }
  if (typeof element !== 'string' || element.length !== 1) return -1
  const code = element.charCodeAt(0)
  return code >= radix ? -1 : code
}

function packingDigits(
  elements: ArrayLike<unknown>,
  radix: number,
  domain: ElementDomain,
): Uint32Array | null {
  const digits = new Uint32Array(elements.length)
  for (let index = 0; index < elements.length; index++) {
    const digit = packingDigit(elements[index], domain, radix)
    if (digit < 0) return null
    digits[index] = digit
  }
  return digits
}

export function packedKeys(
  elements: ArrayLike<unknown>,
  gramSize: number,
  gramCount: number,
  radix: number,
  domain: ElementDomain,
): Float64Array | null {
  if (gramSize === 2) {
    let first = packingDigit(elements[0], domain, radix)
    if (first < 0) return null
    const keys = new Float64Array(gramCount)
    for (let start = 0; start < gramCount; start++) {
      const second = packingDigit(elements[start + 1], domain, radix)
      if (second < 0) return null
      keys[start] = first * radix + second
      first = second
    }
    return keys
  }
  if (gramSize === 3) {
    let first = packingDigit(elements[0], domain, radix)
    if (first < 0) return null
    let second = packingDigit(elements[1], domain, radix)
    if (second < 0) return null
    const keys = new Float64Array(gramCount)
    for (let start = 0; start < gramCount; start++) {
      const third = packingDigit(elements[start + 2], domain, radix)
      if (third < 0) return null
      keys[start] = (first * radix + second) * radix + third
      first = second
      second = third
    }
    return keys
  }
  const digits = packingDigits(elements, radix, domain)
  if (digits === null) return null
  const keys = new Float64Array(gramCount)
  for (let start = 0; start < gramCount; start++) {
    keys[start] = packGram(digits, start, gramSize, radix)
  }
  return keys
}

export function domainOf(elements: ArrayLike<unknown>): ElementDomain {
  return typeof elements[0] === 'string' ? 'char' : 'number'
}

export function domainElement(digit: number, domain: ElementDomain): unknown {
  return domain === 'number' ? digit : String.fromCharCode(digit)
}
