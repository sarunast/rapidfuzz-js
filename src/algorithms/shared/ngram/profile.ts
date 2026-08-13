import type { Sequence } from '../../../core/types.js'
import { convSequence, elementsEqual } from '../sequence.js'
import { canonicalRadix } from './key.js'
import { domainOf, packedKeys, type ElementDomain } from './packing.js'

/**
 * One trie level. A node holds `children` above the last level and `counts` at
 * it — never both, and neither until something is inserted through it. Giving
 * every distinct gram a leaf object with an empty `Map` it could never use cost
 * 750 MiB for 100k prepared bigram profiles against this shape's 331 MiB.
 */
export interface GramNode {
  children: Map<unknown, GramNode> | null
  counts: Map<unknown, number> | null
}

/**
 * The grams as sorted distinct keys with their frequencies, which is what a
 * profile holds whenever every element fits `canonicalRadix(gramSize)`.
 *
 * A trie of `Map`s spends about 228 bytes a gram on an 89-character path — 127
 * `Map` objects for 86 grams — where two typed arrays spend twelve.
 */
export interface PackedGrams {
  readonly kind: 'packed'
  readonly elementDomain: ElementDomain
  // The scale the keys were spelled at, carried rather than recomputed: it is
  // `canonicalRadix(gramSize)` and the packing proves that is not `null`, so a
  // reader that has keys never has to ask again.
  readonly radix: number
  readonly keys: Float64Array
  readonly counts: Uint32Array
}

/** The general representation: any element at all, compared by identity. */
export interface GramTrie {
  readonly kind: 'trie'
  readonly root: GramNode
}

/**
 * One or the other, never both and never neither — a nullable `packed` beside a
 * nullable `root` would spell two states that mean nothing.
 */
export type ProfileStorage = PackedGrams | GramTrie

export class NGramProfile {
  constructor(
    readonly gramSize: number,
    readonly gramCount: number,
    readonly squaredNorm: number,
    readonly storage: ProfileStorage,
    // Retained only when there are no grams, which is the one case a metric has
    // to fall back on comparing the sequences themselves. A profile that has
    // grams would otherwise hold its converted input alive for nothing.
    readonly elements: ArrayLike<unknown> | null,
  ) {}
}

/**
 * The two node shapes a fixed-depth builder knows it is making. Descending
 * through one costs no null check, which is what lets the specialized builders
 * below read a level without `childrenOf`.
 */
interface GramLeaf extends GramNode {
  children: null
  counts: Map<unknown, number>
}

interface GramBranch extends GramNode {
  children: Map<unknown, GramLeaf>
  counts: null
}

export function emptyNode(): GramNode {
  return { children: null, counts: null }
}

function trieStorage(root: GramNode): GramTrie {
  return { kind: 'trie', root }
}

/**
 * How many windows of this depth a sequence holds, never below zero. Each
 * builder derives it rather than being handed it: a count and the elements it
 * counts are one invariant, and passing them separately lets them disagree.
 */
export function gramsIn(elements: ArrayLike<unknown>, gramSize: number): number {
  const gramCount = elements.length - gramSize + 1
  return gramCount > 0 ? gramCount : 0
}

function leafNode(): GramLeaf {
  return { children: null, counts: new Map<unknown, number>() }
}

// `NaN !== NaN`, which is the element equality every other metric here uses,
// while a `Map` keyed by one matches it under SameValueZero. A gram holding
// `NaN` is therefore unmatchable and is never inserted — it still counts toward
// `gramCount` and `squaredNorm`, so denominators and search bounds stay right.
function isUnmatchable(element: unknown): boolean {
  return typeof element === 'number' && Number.isNaN(element)
}

/**
 * Consecutive grams overlap in every element but one, so a fixed depth carries
 * the window forward and reads each element once where the generic loop
 * re-reads it `gramSize` times — and builds each node in its final shape
 * rather than assigning into an empty one.
 *
 * Worth 1.05-1.15x on a direct comparison, growing with length, and about 1.2x
 * on 4096 characters of one repeated gram. The profiles it produces are
 * identical to the byte at 100k retained.
 */
function bigramProfile(elements: ArrayLike<unknown>, gramCount: number): NGramProfile {
  const children = new Map<unknown, GramLeaf>()
  const root: GramBranch = { children, counts: null }
  let first = elements[0]
  let lastUnmatchable = isUnmatchable(first) ? 0 : -1
  let squaredNorm = 0
  for (let start = 0; start < gramCount; start++) {
    const end = start + 1
    const second = elements[end]
    if (isUnmatchable(second)) lastUnmatchable = end
    if (lastUnmatchable >= start) {
      squaredNorm++
      first = second
      continue
    }
    let child = children.get(first)
    if (child === undefined) {
      child = leafNode()
      children.set(first, child)
    }
    const counts = child.counts
    const count = counts.get(second) ?? 0
    squaredNorm += 2 * count + 1
    counts.set(second, count + 1)
    first = second
  }
  return new NGramProfile(2, gramCount, squaredNorm, trieStorage(root), null)
}

function trigramProfile(elements: ArrayLike<unknown>, gramCount: number): NGramProfile {
  const children = new Map<unknown, GramBranch>()
  const root: GramNode = { children, counts: null }
  let first = elements[0]
  let second = elements[1]
  let lastUnmatchable = isUnmatchable(second) ? 1 : isUnmatchable(first) ? 0 : -1
  let squaredNorm = 0
  for (let start = 0; start < gramCount; start++) {
    const end = start + 2
    const third = elements[end]
    if (isUnmatchable(third)) lastUnmatchable = end
    if (lastUnmatchable >= start) {
      squaredNorm++
      first = second
      second = third
      continue
    }
    let child = children.get(first)
    if (child === undefined) {
      child = { children: new Map<unknown, GramLeaf>(), counts: null }
      children.set(first, child)
    }
    const level = child.children
    let grandchild = level.get(second)
    if (grandchild === undefined) {
      grandchild = leafNode()
      level.set(second, grandchild)
    }
    const counts = grandchild.counts
    const count = counts.get(third) ?? 0
    squaredNorm += 2 * count + 1
    counts.set(third, count + 1)
    first = second
    second = third
  }
  return new NGramProfile(3, gramCount, squaredNorm, trieStorage(root), null)
}

/**
 * The grams as sorted distinct keys, or `null` where the input cannot be packed
 * — an object element, a negative, `NaN`, an element past the canonical radix,
 * a mixed domain, or a `gramSize` with no rung at all.
 *
 * Exported because it is half of {@link profileOfElements}, and because the
 * differential tests compare it against `trieProfile` over the same input.
 */
export function packedProfile(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile | null {
  const gramCount = gramsIn(elements, gramSize)
  // No gram is no element domain to establish, and such a profile has to keep
  // its elements for `zeroGramSimilarity` — both of which are the trie's job.
  if (gramCount === 0) return null
  const radix = canonicalRadix(gramSize)
  if (radix === null) return null
  const domain = domainOf(elements)

  // Sorting every gram, rather than tallying into a `Map` and sorting only the
  // distinct keys. The `Map` shape wins only where a long sequence draws on a
  // tiny alphabet — 0.64x on 4096 characters of bigrams over 26 letters, all 676
  // of which repeat — and loses 1.56-2.63x everywhere else, including the shapes
  // prepared search is made of. The sort is 87% of this build at 4096
  // characters, so that one case is the price of the other twenty-eight.
  const sorted = packedKeys(elements, gramSize, gramCount, radix, domain)
  if (sorted === null) return null
  sorted.sort()

  let distinct = 0
  for (let index = 0; index < gramCount; index++) {
    if (index === 0 || sorted[index] !== sorted[index - 1]) distinct++
  }
  // Sized exactly rather than sliced from a scratch buffer: a subarray would
  // retain the whole `gramCount`-wide allocation behind every profile, which is
  // the cost this representation exists to remove.
  const keys = new Float64Array(distinct)
  const counts = new Uint32Array(distinct)
  let at = -1
  for (let index = 0; index < gramCount; index++) {
    const key = sorted[index]
    if (at < 0 || key !== keys[at]) {
      at++
      keys[at] = key
      counts[at] = 1
      continue
    }
    counts[at]++
  }
  let squaredNorm = 0
  for (let index = 0; index < distinct; index++)
    squaredNorm += counts[index] * counts[index]

  return new NGramProfile(
    gramSize,
    gramCount,
    squaredNorm,
    { kind: 'packed', elementDomain: domain, radix, keys, counts },
    null,
  )
}

/**
 * An exact multiset of a sequence's n-grams, as a trie of depth `gramSize`.
 *
 * A trie rather than a serialized key because elements are arbitrary values:
 * `['a,b', 'c']` and `['a', 'b,c']` are different grams that any separator
 * would collide, and `JSON.stringify` cannot see object identity at all.
 *
 * Exported for the differential tests, which need to build this representation
 * for an input {@link packedProfile} would have taken.
 */
export function trieProfile(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile {
  const gramCount = gramsIn(elements, gramSize)
  // A sequence shorter than one gram keeps its elements, because comparing them
  // directly is all `zeroGramSimilarity` has left to do. The invariant lives
  // here rather than in `profileOfElements` so that forcing a representation
  // and letting one be chosen answer the same thing.
  if (gramCount === 0) {
    return new NGramProfile(gramSize, 0, 0, trieStorage(emptyNode()), elements)
  }
  if (gramSize === 2) return bigramProfile(elements, gramCount)
  if (gramSize === 3) return trigramProfile(elements, gramCount)

  const root = emptyNode()
  // Each element is tested once rather than once per window it appears in, so
  // a gram is skipped by comparing its start against the last one seen.
  let lastUnmatchable = -1
  for (let index = 0; index < gramSize - 1; index++) {
    if (isUnmatchable(elements[index])) lastUnmatchable = index
  }
  // `Σ count²`, maintained as each gram arrives: `(c + 1)² - c²` is `2c + 1`,
  // which is cheaper than a second pass over the finished trie.
  let squaredNorm = 0
  const last = gramSize - 1
  for (let start = 0; start < gramCount; start++) {
    const end = start + last
    if (isUnmatchable(elements[end])) lastUnmatchable = end
    if (lastUnmatchable >= start) {
      squaredNorm++
      continue
    }
    let node = root
    for (let offset = 0; offset < last; offset++) {
      const element = elements[start + offset]
      let children = node.children
      if (children === null) {
        children = new Map<unknown, GramNode>()
        node.children = children
      }
      let child = children.get(element)
      if (child === undefined) {
        child = emptyNode()
        children.set(element, child)
      }
      node = child
    }
    let counts = node.counts
    if (counts === null) {
      counts = new Map<unknown, number>()
      node.counts = counts
    }
    const element = elements[end]
    const count = counts.get(element) ?? 0
    squaredNorm += 2 * count + 1
    counts.set(element, count + 1)
  }
  return new NGramProfile(gramSize, gramCount, squaredNorm, trieStorage(root), null)
}

/**
 * The multiset a metric compares, packed where the elements allow it.
 *
 * `gramSize` is trusted here, and validated once by the caller: `validGramSize`
 * on the direct path, `parseGramSize` when a scorer compiles. Re-checking it
 * would put the check inside the loop that builds every prepared choice.
 *
 * A sequence with no grams keeps trie storage and its elements: there is no
 * gram to establish an element domain from, and `zeroGramSimilarity` compares
 * the sequences themselves.
 */
export function profileOfElements(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile {
  return packedProfile(elements, gramSize) ?? trieProfile(elements, gramSize)
}

export function buildProfile(sequence: Sequence, gramSize: number): NGramProfile {
  return profileOfElements(convSequence(sequence), gramSize)
}

export function preparedProfile(value: unknown): NGramProfile {
  if (!(value instanceof NGramProfile)) {
    throw new TypeError('invalid prepared n-gram profile')
  }
  return value
}

// One empty map of each kind, shared by every node that has no children or no
// counts, so a reader never branches on null and no branch exists that only one
// trie depth could reach. `ReadonlyMap` is what keeps that safe across the
// subsystem: a caller reaching one of these sentinels and writing to it would
// corrupt every profile at once, and every reader outside this file only
// iterates. Insertion goes through the node's own field, never the accessor.
const NO_CHILDREN: ReadonlyMap<unknown, GramNode> = /* @__PURE__ */ new Map()
const NO_COUNTS: ReadonlyMap<unknown, number> = /* @__PURE__ */ new Map()

export function childrenOf(node: GramNode): ReadonlyMap<unknown, GramNode> {
  return node.children ?? NO_CHILDREN
}

export function countsOf(node: GramNode): ReadonlyMap<unknown, number> {
  return node.counts ?? NO_COUNTS
}

/**
 * A profile with no grams keeps its elements, and only such a profile does, so
 * the retained sequence is itself the "compare these directly" signal.
 */
export function zeroGramSimilarity(a: NGramProfile, b: NGramProfile): number {
  const elementsA = a.elements
  const elementsB = b.elements
  return elementsA !== null && elementsB !== null && elementsEqual(elementsA, elementsB)
    ? 1
    : 0
}
