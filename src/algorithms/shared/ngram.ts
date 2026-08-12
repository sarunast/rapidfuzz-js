import type { Sequence } from '../../core/types.js'
import { convSequence } from './sequence.js'

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

export class NGramProfile {
  constructor(
    readonly gramSize: number,
    readonly gramCount: number,
    readonly squaredNorm: number,
    readonly root: GramNode,
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

function emptyNode(): GramNode {
  return { children: null, counts: null }
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
 * Worth 1.05-1.15x on a direct comparison, growing with length, and about
 * 1.2x on 4096 characters of one repeated gram. Measured against the previous
 * build in a separate process each, ratioed against `dice-coefficient` in that
 * same process so no drift between the two runs survives: `bench:compare` put
 * this at 1.25x, and re-running the *old* code against the *old* baseline
 * reproduced 0.67-0.92x on those same cases, so the instrument was the
 * artefact. Profiles are identical to the byte at 100k retained.
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
  return new NGramProfile(2, gramCount, squaredNorm, root, null)
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
  return new NGramProfile(3, gramCount, squaredNorm, root, null)
}

/**
 * An exact multiset of a sequence's n-grams, as a trie of depth `gramSize`.
 *
 * A trie rather than a serialized key because elements are arbitrary values:
 * `['a,b', 'c']` and `['a', 'b,c']` are different grams that any separator
 * would collide, and `JSON.stringify` cannot see object identity at all.
 *
 * `gramSize` is trusted here, and validated once above: `validGramSize` on the
 * direct path, `parseGramSize` when a scorer compiles. Re-checking it would put
 * the check inside the loop that builds every prepared choice.
 */
export function profileOfElements(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile {
  const gramCount = elements.length - gramSize + 1
  if (gramCount <= 0) {
    return new NGramProfile(gramSize, 0, 0, emptyNode(), elements)
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
  return new NGramProfile(gramSize, gramCount, squaredNorm, root, null)
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
// trie depth could reach. Never written to: `profileOfElements` installs a fresh
// map into the node before it inserts anything.
const NO_CHILDREN: Map<unknown, GramNode> = /* @__PURE__ */ new Map()
const NO_COUNTS: Map<unknown, number> = /* @__PURE__ */ new Map()

function childrenOf(node: GramNode): Map<unknown, GramNode> {
  return node.children ?? NO_CHILDREN
}

function countsOf(node: GramNode): Map<unknown, number> {
  return node.counts ?? NO_COUNTS
}

function sharedCounts(a: GramNode, b: GramNode): number {
  const countsB = countsOf(b)
  let shared = 0
  for (const [element, countA] of countsOf(a)) {
    const countB = countsB.get(element)
    if (countB !== undefined) shared += Math.min(countA, countB)
  }
  return shared
}

function dotCounts(a: GramNode, b: GramNode): number {
  const countsB = countsOf(b)
  let product = 0
  for (const [element, countA] of countsOf(a)) {
    const countB = countsB.get(element)
    if (countB !== undefined) product += countA * countB
  }
  return product
}

/**
 * `Σ min(a_g, b_g)` over the grams the two profiles share.
 *
 * The three depths a caller asks for get literal loops. The generic walk
 * allocates three stack arrays per comparison, which measured 1.6x the
 * specialized bigram loop over 100 queries against 1000 prepared choices — and
 * over prebuilt trigram profiles, at four lengths from 12 to 512 characters,
 * 1.2-1.7x the literal trigram loop below.
 *
 * Deeper than that it is iterative over an explicit stack, not recursive:
 * `gramSize` is caller-supplied and equals the trie depth, so recursion would
 * put a stack overflow inside the range of valid inputs.
 */
export function sharedFrequency(a: NGramProfile, b: NGramProfile): number {
  const gramSize = a.gramSize
  if (gramSize === 1) return sharedCounts(a.root, b.root)
  if (gramSize === 2) {
    const childrenB = childrenOf(b.root)
    let shared = 0
    for (const [element, childA] of childrenOf(a.root)) {
      const childB = childrenB.get(element)
      if (childB !== undefined) shared += sharedCounts(childA, childB)
    }
    return shared
  }
  if (gramSize === 3) {
    const childrenB = childrenOf(b.root)
    let shared = 0
    for (const [first, childA] of childrenOf(a.root)) {
      const childB = childrenB.get(first)
      if (childB === undefined) continue
      const levelB = childrenOf(childB)
      for (const [second, grandchildA] of childrenOf(childA)) {
        const grandchildB = levelB.get(second)
        if (grandchildB !== undefined) shared += sharedCounts(grandchildA, grandchildB)
      }
    }
    return shared
  }
  let shared = 0
  const last = gramSize - 1
  const nodesA: GramNode[] = [a.root]
  const nodesB: GramNode[] = [b.root]
  const depths: number[] = [0]
  let top = 1
  while (top > 0) {
    top--
    const nodeA = nodesA[top]
    const nodeB = nodesB[top]
    const depth = depths[top]
    if (depth === last) {
      shared += sharedCounts(nodeA, nodeB)
      continue
    }
    const levelB = childrenOf(nodeB)
    for (const [element, childA] of childrenOf(nodeA)) {
      const childB = levelB.get(element)
      if (childB !== undefined) {
        nodesA[top] = childA
        nodesB[top] = childB
        depths[top] = depth + 1
        top++
      }
    }
  }
  return shared
}

/**
 * `Σ a_g · b_g` over the grams the two profiles share.
 *
 * A second literal traversal rather than {@link sharedFrequency} taking a
 * combiner: the innermost frame of a walk is the last place to put a callback
 * every n-gram metric would make megamorphic.
 */
export function dotProduct(a: NGramProfile, b: NGramProfile): number {
  const gramSize = a.gramSize
  if (gramSize === 1) return dotCounts(a.root, b.root)
  if (gramSize === 2) {
    const childrenB = childrenOf(b.root)
    let product = 0
    for (const [element, childA] of childrenOf(a.root)) {
      const childB = childrenB.get(element)
      if (childB !== undefined) product += dotCounts(childA, childB)
    }
    return product
  }
  if (gramSize === 3) {
    const childrenB = childrenOf(b.root)
    let product = 0
    for (const [first, childA] of childrenOf(a.root)) {
      const childB = childrenB.get(first)
      if (childB === undefined) continue
      const levelB = childrenOf(childB)
      for (const [second, grandchildA] of childrenOf(childA)) {
        const grandchildB = levelB.get(second)
        if (grandchildB !== undefined) product += dotCounts(grandchildA, grandchildB)
      }
    }
    return product
  }
  let product = 0
  const last = gramSize - 1
  const nodesA: GramNode[] = [a.root]
  const nodesB: GramNode[] = [b.root]
  const depths: number[] = [0]
  let top = 1
  while (top > 0) {
    top--
    const nodeA = nodesA[top]
    const nodeB = nodesB[top]
    const depth = depths[top]
    if (depth === last) {
      product += dotCounts(nodeA, nodeB)
      continue
    }
    const levelB = childrenOf(nodeB)
    for (const [element, childA] of childrenOf(nodeA)) {
      const childB = levelB.get(element)
      if (childB !== undefined) {
        nodesA[top] = childA
        nodesB[top] = childB
        depths[top] = depth + 1
        top++
      }
    }
  }
  return product
}

export function elementsEqual(a: ArrayLike<unknown>, b: ArrayLike<unknown>): boolean {
  if (a.length !== b.length) return false
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false
  }
  return true
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

/**
 * `isSafeInteger` rather than `isInteger`: `1e300` is an integer, and a trie
 * that deep is not a request anyone means. Testing for the valid range rather
 * than against it rejects `NaN` with the same comparison.
 */
export function validGramSize(value: unknown): number {
  if (value == null) return 2
  if (typeof value !== 'number') throw new TypeError('gramSize must be a number')
  if (!(Number.isSafeInteger(value) && value >= 1)) {
    throw new RangeError('gramSize has to be a safe integer of at least 1')
  }
  return value
}

export function parseGramSize(options: Readonly<Record<string, unknown>>): number {
  return validGramSize(Reflect.get(options, 'gramSize'))
}

/**
 * An intersection against one fixed query profile.
 *
 * A query does not change while a search runs, so its trie is walked once, at
 * preparation, into flat arrays — and every candidate after that is indexed
 * loops over those plus a `Map.get` per gram. Measured over 100 queries against
 * 1000 prepared bigram choices, at four length pairs: 0.69-0.74x the cost of
 * walking both tries per candidate.
 *
 * Flattened up to `gramSize` 3, which is where the depths a caller actually
 * asks for stop: the trigram kernel measured 0.48x (12-char) and 0.62x
 * (32-char) against the generic walk over the same 100x1000 shape, because that
 * walk allocates three stack arrays for every candidate. Deeper stays generic —
 * a fourth level of nested key arrays buys a case nobody has asked for.
 */
export interface FrequencyKernel {
  (choice: NGramProfile): number
}

/**
 * A kernel that may stop early once `minimumShared` is out of reach, returning
 * some count below it rather than the true intersection. Only a caller that
 * would reject anything smaller may pass one — see `sharedFrequencyKernel`.
 */
export interface BoundedFrequencyKernel {
  (choice: NGramProfile, minimumShared: number): number
}

interface FlatLevel {
  readonly firstKeys: unknown[]
  readonly secondKeys: unknown[][]
  readonly frequencies: number[][]
  readonly remaining: Uint32Array
}

/**
 * `remaining[index]` is the frequency still to come from group `index` onward,
 * so `shared + remaining[index]` is everything the walk could still reach.
 * Summed from the inserted counts and never from `gramCount`: a gram holding
 * `NaN` counts toward the latter and can never be matched.
 *
 * Unsigned because the total is bounded by the sequence length, which the
 * contract allows up to `0xffff_ffff`: signed, a query past 2^31 grams stores a
 * negative bound and the walk rejects candidates that qualify.
 */
function remainingTotals(totals: readonly number[]): Uint32Array {
  const remaining = new Uint32Array(totals.length + 1)
  for (let index = totals.length - 1; index >= 0; index--) {
    remaining[index] = remaining[index + 1] + totals[index]
  }
  return remaining
}

function flattenUnigrams(query: NGramProfile): [unknown[], number[], Uint32Array] {
  const keys: unknown[] = []
  const frequencies: number[] = []
  for (const [element, count] of countsOf(query.root)) {
    keys.push(element)
    frequencies.push(count)
  }
  return [keys, frequencies, remainingTotals(frequencies)]
}

interface FlatTrigramLevel {
  readonly firstKeys: unknown[]
  readonly secondKeys: unknown[][]
  readonly thirdKeys: unknown[][][]
  readonly frequencies: number[][][]
  readonly remaining: Uint32Array
}

function flattenBigrams(query: NGramProfile): FlatLevel {
  const firstKeys: unknown[] = []
  const secondKeys: unknown[][] = []
  const frequencies: number[][] = []
  const totals: number[] = []
  for (const [first, child] of childrenOf(query.root)) {
    const keys: unknown[] = []
    const counts: number[] = []
    let total = 0
    for (const [second, count] of countsOf(child)) {
      keys.push(second)
      counts.push(count)
      total += count
    }
    firstKeys.push(first)
    secondKeys.push(keys)
    frequencies.push(counts)
    totals.push(total)
  }
  return { firstKeys, secondKeys, frequencies, remaining: remainingTotals(totals) }
}

function flattenTrigrams(query: NGramProfile): FlatTrigramLevel {
  const firstKeys: unknown[] = []
  const secondKeys: unknown[][] = []
  const thirdKeys: unknown[][][] = []
  const frequencies: number[][][] = []
  const totals: number[] = []
  for (const [first, child] of childrenOf(query.root)) {
    const seconds: unknown[] = []
    const thirds: unknown[][] = []
    const counts: number[][] = []
    let total = 0
    for (const [second, grandchild] of childrenOf(child)) {
      const keys: unknown[] = []
      const values: number[] = []
      for (const [third, count] of countsOf(grandchild)) {
        keys.push(third)
        values.push(count)
        total += count
      }
      seconds.push(second)
      thirds.push(keys)
      counts.push(values)
    }
    firstKeys.push(first)
    secondKeys.push(seconds)
    thirdKeys.push(thirds)
    frequencies.push(counts)
    totals.push(total)
  }
  return {
    firstKeys,
    secondKeys,
    thirdKeys,
    frequencies,
    remaining: remainingTotals(totals),
  }
}

/**
 * The intersection above, plus the one bound the gram counts cannot express:
 * candidates of the query's own length have an upper bound of 1 however little
 * they share, so only what the walk has found so far can turn them down.
 * `shared + remaining[index]` is everything still reachable, and a rising search
 * cutoff makes that fail sooner the longer the search runs.
 *
 * `bench/ngram.bench.ts` over 2000 same-length candidates, comparing against a
 * baseline recorded minutes earlier rather than a stored one: 1.11-1.20x over
 * four runs on `search` with a limit and a threshold, and nothing worse
 * anywhere — the `bounded` flag exists for that second half, since an unlimited
 * search asks for no minimum and must not pay a load per group to be told so.
 * `bestMatch` over the same candidates gains 0-7%: it starts at no cutoff and
 * only raises one as it goes, so the bound arrives late.
 */
export function sharedFrequencyKernel(query: NGramProfile): BoundedFrequencyKernel {
  const gramSize = query.gramSize
  if (gramSize === 1) {
    const [keys, frequencies, remaining] = flattenUnigrams(query)
    return (choice, minimumShared) => {
      const counts = countsOf(choice.root)
      const bounded = minimumShared > 0
      let shared = 0
      for (let index = 0; index < keys.length; index++) {
        if (bounded && shared + remaining[index] < minimumShared) return shared
        const other = counts.get(keys[index])
        if (other !== undefined) {
          const mine = frequencies[index]
          shared += mine < other ? mine : other
        }
      }
      return shared
    }
  }
  if (gramSize === 2) {
    const { firstKeys, secondKeys, frequencies, remaining } = flattenBigrams(query)
    return (choice, minimumShared) => {
      const children = childrenOf(choice.root)
      const bounded = minimumShared > 0
      let shared = 0
      for (let index = 0; index < firstKeys.length; index++) {
        if (bounded && shared + remaining[index] < minimumShared) return shared
        const child = children.get(firstKeys[index])
        if (child === undefined) continue
        const counts = countsOf(child)
        const keys = secondKeys[index]
        const mine = frequencies[index]
        for (let inner = 0; inner < keys.length; inner++) {
          const other = counts.get(keys[inner])
          if (other !== undefined) {
            const count = mine[inner]
            shared += count < other ? count : other
          }
        }
      }
      return shared
    }
  }
  if (gramSize === 3) {
    const { firstKeys, secondKeys, thirdKeys, frequencies, remaining } =
      flattenTrigrams(query)
    return (choice, minimumShared) => {
      const children = childrenOf(choice.root)
      const bounded = minimumShared > 0
      let shared = 0
      for (let index = 0; index < firstKeys.length; index++) {
        if (bounded && shared + remaining[index] < minimumShared) return shared
        const child = children.get(firstKeys[index])
        if (child === undefined) continue
        const level = childrenOf(child)
        const seconds = secondKeys[index]
        const thirds = thirdKeys[index]
        const mine = frequencies[index]
        for (let inner = 0; inner < seconds.length; inner++) {
          const grandchild = level.get(seconds[inner])
          if (grandchild === undefined) continue
          const counts = countsOf(grandchild)
          const keys = thirds[inner]
          const values = mine[inner]
          for (let leaf = 0; leaf < keys.length; leaf++) {
            const other = counts.get(keys[leaf])
            if (other !== undefined) {
              const count = values[leaf]
              shared += count < other ? count : other
            }
          }
        }
      }
      return shared
    }
  }
  return (choice) => sharedFrequency(query, choice)
}

export function dotProductKernel(query: NGramProfile): FrequencyKernel {
  const gramSize = query.gramSize
  if (gramSize === 1) {
    const [keys, frequencies] = flattenUnigrams(query)
    return (choice) => {
      const counts = countsOf(choice.root)
      let product = 0
      for (let index = 0; index < keys.length; index++) {
        const other = counts.get(keys[index])
        if (other !== undefined) product += frequencies[index] * other
      }
      return product
    }
  }
  if (gramSize === 2) {
    const { firstKeys, secondKeys, frequencies } = flattenBigrams(query)
    return (choice) => {
      const children = childrenOf(choice.root)
      let product = 0
      for (let index = 0; index < firstKeys.length; index++) {
        const child = children.get(firstKeys[index])
        if (child === undefined) continue
        const counts = countsOf(child)
        const keys = secondKeys[index]
        const mine = frequencies[index]
        for (let inner = 0; inner < keys.length; inner++) {
          const other = counts.get(keys[inner])
          if (other !== undefined) product += mine[inner] * other
        }
      }
      return product
    }
  }
  if (gramSize === 3) {
    const { firstKeys, secondKeys, thirdKeys, frequencies } = flattenTrigrams(query)
    return (choice) => {
      const children = childrenOf(choice.root)
      let product = 0
      for (let index = 0; index < firstKeys.length; index++) {
        const child = children.get(firstKeys[index])
        if (child === undefined) continue
        const level = childrenOf(child)
        const seconds = secondKeys[index]
        const thirds = thirdKeys[index]
        const mine = frequencies[index]
        for (let inner = 0; inner < seconds.length; inner++) {
          const grandchild = level.get(seconds[inner])
          if (grandchild === undefined) continue
          const counts = countsOf(grandchild)
          const keys = thirds[inner]
          const values = mine[inner]
          for (let leaf = 0; leaf < keys.length; leaf++) {
            const other = counts.get(keys[leaf])
            if (other !== undefined) product += values[leaf] * other
          }
        }
      }
      return product
    }
  }
  return (choice) => dotProduct(query, choice)
}
