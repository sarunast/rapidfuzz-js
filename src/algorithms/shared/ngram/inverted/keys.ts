import { feasibleRadices, packGram, unpackGram } from '../key.js'

/**
 * Carries both the element that did not fit and the radix it did not fit — the
 * second so the re-key needs no null check for a state it cannot be in. Only a
 * packed radix can raise this, and the error itself is the proof.
 */
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

/**
 * The narrowest feasible radix holding `element`, or `null` for joined strings.
 *
 * A negative element goes straight to strings: positional packing has no room
 * below zero, so answering with a rung the element is merely *less than* would
 * hand the re-key a target no wider than the one that just failed, and the
 * ladder would report it could not widen on an element strings represent
 * exactly.
 */
export function radixFor(gramSize: number, element: number): number | null {
  if (element < 0) return null
  for (const radix of feasibleRadices(gramSize)) if (element < radix) return radix
  return null
}

/**
 * The same gram, re-spelled for a wider radix or for joined strings. Packing is
 * positional and so reversible, which is what lets an index that has already
 * ingested a million choices change key scheme without re-reading one of them.
 *
 * A key that is already joined comes back unchanged. No build reaches that arm —
 * only a packed radix can raise the error that starts a re-key, and once the
 * scheme is joined nothing raises it again — so the case is pinned by a direct
 * test rather than by a corpus.
 */
export function repackKey(
  key: string | number,
  from: number,
  to: number | null,
  gramSize: number,
): string | number {
  if (typeof key === 'string') return key
  // Through the shared primitives rather than open-coded: this is the same
  // positional arithmetic `key.ts` owns, and re-keying is an exceptional
  // corpus-build event rather than a scoring loop, so there is nothing to buy
  // by spelling it twice.
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

/**
 * Every distinct gram of a sequence, with its frequency, written into
 * caller-owned arrays.
 *
 * One walk shared by indexing and by querying, so the two cannot drift apart on
 * how a gram becomes a key — the only way an index could disagree with the
 * metric it reproduces. `widening` is the difference between them: during a
 * build an element too wide for the current radix has to widen the whole index,
 * while against an already-packed one a query spells that gram as a joined key
 * instead — and a packed index holds no string keys, so it matches nothing.
 * That is the overflow case only. An index whose own scheme is already joined
 * (`radix === null`) keys both sides the same way, and they match normally.
 *
 * That fallback is not the same as dropping the gram, and the difference is
 * Cosine's denominator: an unmatchable gram still counts toward the query's own
 * norm, so it has to reach `counts` even though no posting list will name it.
 *
 * Returns the squared norm, since counting is where it comes from for free —
 * `Σ c²` accumulates as `2·previous + 1` per occurrence.
 */
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
