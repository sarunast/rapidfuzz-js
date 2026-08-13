import { packGram } from './key.js'

/**
 * Which domain a packed profile's elements came from. `'a' !== 97` here, and a
 * key of `97` could mean either, so the domain travels with the keys and two
 * profiles that disagree on it share nothing.
 */
export type ElementDomain = 'number' | 'char'

/**
 * One element's digit, or `-1` where this domain and radix cannot spell it.
 *
 * `isInteger` rejects `NaN` and both infinities with the same comparison, and
 * the range check rejects a negative, which positional packing has no room for.
 */
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

/**
 * Every element as its packing digit, or `null` for a sequence that has to stay
 * a trie.
 *
 * The domain comes from the first element and every later one has to agree:
 * `[97, 'b']` is packable twice over on its own terms, and packing it would
 * make `'b'` and `98` the same gram where the trie keeps them apart. A caller
 * that converts its input never produces such a sequence; this function is
 * given arbitrary ones.
 */
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

/**
 * Every gram's packed key, in sequence order, or `null` for a sequence that has
 * to stay a trie.
 *
 * Depths 2 and 3 carry the window forward instead of re-reading it: successive
 * grams overlap in all but one element, so a rolling form reads each element
 * once and needs no digit array at all, where the generic path reads every
 * element `gramSize` times through one. Worth 1.06-1.55x over the whole build,
 * most of it at short inputs where the dropped allocation is the call.
 */
export function packedKeys(
  elements: ArrayLike<unknown>,
  gramSize: number,
  gramCount: number,
  radix: number,
  domain: ElementDomain,
): Float64Array | null {
  // Nothing is allocated until the elements the window is seeded from have
  // proved packable: a sequence that cannot be packed at all is the fallback
  // path, and it should not first buy a buffer for the answer it will not give.
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
  // The generic path validates every element before any of them is packed, so
  // this allocation is reached only for a sequence that is going to fill it.
  const digits = packingDigits(elements, radix, domain)
  if (digits === null) return null
  const keys = new Float64Array(gramCount)
  for (let start = 0; start < gramCount; start++) {
    keys[start] = packGram(digits, start, gramSize, radix)
  }
  return keys
}

/** The domain a sequence's first element puts it in, which every later one has to share. */
export function domainOf(elements: ArrayLike<unknown>): ElementDomain {
  return typeof elements[0] === 'string' ? 'char' : 'number'
}

/**
 * The element a packed digit stands for, which is what a trie is keyed by.
 * Decoding to the digit alone would have `'a'` miss the `'a'` a trie holds.
 */
export function domainElement(digit: number, domain: ElementDomain): unknown {
  return domain === 'number' ? digit : String.fromCharCode(digit)
}
