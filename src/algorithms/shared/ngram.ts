import type { Sequence } from '../../core/types.js'
import { canonicalRadix, packGram, unpackGram } from './gramKey.js'
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

/**
 * Which domain a packed profile's elements came from. `'a' !== 97` here, and a
 * key of `97` could mean either, so the domain travels with the keys and two
 * profiles that disagree on it share nothing.
 */
export type ElementDomain = 'number' | 'char'

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

function emptyNode(): GramNode {
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
function gramsIn(elements: ArrayLike<unknown>, gramSize: number): number {
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
 * One element's digit, or `-1` where this domain and radix cannot spell it.
 *
 * `isInteger` rejects `NaN` and both infinities with the same comparison, and
 * the range check rejects a negative, which positional packing has no room for.
 */
function packingDigit(element: unknown, domain: ElementDomain, radix: number): number {
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
  const first = elements[0]
  const domain: ElementDomain = typeof first === 'string' ? 'char' : 'number'
  const digits = packingDigits(elements, radix, domain)
  if (digits === null) return null

  // Sorting every gram, rather than tallying into a `Map` and sorting only the
  // distinct keys. The `Map` shape wins exactly where a long sequence draws on
  // a tiny alphabet — 0.64x on 4096 characters of bigrams over 26 letters, all
  // 676 of which repeat — and loses 1.56-2.63x everywhere else, including the
  // shapes prepared search is made of. The sort is 87% of this build (0.208 ms
  // of 0.239 ms at 4096 characters), which is why `bench:confirm` puts that one
  // case at 0.74x while the suite moves +28%.
  const sorted = new Float64Array(gramCount)
  for (let start = 0; start < gramCount; start++) {
    sorted[start] = packGram(digits, start, gramSize, radix)
  }
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
 * `gramSize` is trusted here, and validated once above: `validGramSize` on the
 * direct path, `parseGramSize` when a scorer compiles. Re-checking it would put
 * the check inside the loop that builds every prepared choice.
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
 * The element a packed digit stands for, which is what a trie is keyed by.
 * Decoding to the digit alone would have `'a'` miss the `'a'` a trie holds.
 */
function domainElement(digit: number, domain: ElementDomain): unknown {
  return domain === 'number' ? digit : String.fromCharCode(digit)
}

/**
 * A packed side against a trie one: every packed gram decoded back to its
 * elements and looked up.
 *
 * One function for both operations, and one orientation for both operand
 * orders, because `Σ min` and `Σ a·b` are each symmetric. The trie walks below
 * are literal per depth on a recorded measurement; this path is the rare one —
 * it needs a sequence packing refused, which for converted text means an astral
 * code point at trigram depth — and gets no such licence.
 */
function packedAgainstTrie(
  packed: PackedGrams,
  root: GramNode,
  gramSize: number,
  multiply: boolean,
): number {
  const radix = packed.radix
  const keys = packed.keys
  const counts = packed.counts
  const domain = packed.elementDomain
  const digits = new Array<number>(gramSize)
  const last = gramSize - 1
  let total = 0
  for (let index = 0; index < keys.length; index++) {
    unpackGram(keys[index], gramSize, radix, digits)
    let node = root
    let reached = true
    for (let offset = 0; offset < last; offset++) {
      const child = childrenOf(node).get(domainElement(digits[offset], domain))
      if (child === undefined) {
        reached = false
        break
      }
      node = child
    }
    if (!reached) continue
    const other = countsOf(node).get(domainElement(digits[last], domain))
    if (other === undefined) continue
    const mine = counts[index]
    total += multiply ? mine * other : mine < other ? mine : other
  }
  return total
}

/**
 * How much longer the other side has to be before searching into it beats
 * walking it. Swept over query lengths 5 to 50 against length ratios 1 to 64,
 * both walks over the same arrays: a probe is 0.13-0.44x of a merge from this
 * ratio upward and inside noise of it below, while the ratios it declines — 1
 * to 4 — are where a merge is up to 1.15x ahead. The corner it exists for is a
 * short query against long choices, where a merge measured 3.0x *slower* than
 * the trie kernel it replaces on 5 grams into 500.
 *
 * Swept unbounded, and that is the case that decides it: under a threshold's
 * minimum the bound cuts so early that every shape from 5x5 to 50x1600 costs
 * 0.0055-0.0084 ms whichever arm runs. The arm matters only where the bound
 * does not bite.
 */
const PROBE_LENGTH_RATIO = 8

/**
 * Two packed sides of the same domain and depth.
 *
 * `remaining` is the driving side's suffix totals and turns the walk bounded,
 * which pins that side down; unbounded, both operations are symmetric and the
 * shorter side drives. Either way the driver ascends, so a probe can start each
 * search where the last one landed.
 */
function packedIntersect(
  query: PackedGrams,
  choice: PackedGrams,
  multiply: boolean,
  remaining: Uint32Array | null,
  minimumShared: number,
): number {
  const swap = remaining === null && choice.keys.length < query.keys.length
  const keys = swap ? choice.keys : query.keys
  const counts = swap ? choice.counts : query.counts
  const otherKeys = swap ? query.keys : choice.keys
  const otherCounts = swap ? query.counts : choice.counts
  const bounded = remaining !== null && minimumShared > 0
  let total = 0
  if (otherKeys.length >= keys.length * PROBE_LENGTH_RATIO) {
    let low = 0
    for (let index = 0; index < keys.length; index++) {
      if (bounded && total + remaining[index] < minimumShared) return total
      const key = keys[index]
      let high = otherKeys.length - 1
      let found = -1
      while (low <= high) {
        // The span is shifted rather than the sum: a sequence may hold up to
        // `0xffff_ffff` elements, so `low + high` can pass 2^32 and wrap to a
        // midpoint outside the window, while `high - low` is bounded by the
        // length and `>>>` is exact over it. Shifting matters — the arithmetic
        // is the loop, and `Math.floor((high - low) / 2)` measured 4.3x slower
        // over 5 x 500 keys.
        const middle = low + ((high - low) >>> 1)
        const candidate = otherKeys[middle]
        if (candidate === key) {
          found = middle
          break
        }
        if (candidate < key) low = middle + 1
        else high = middle - 1
      }
      if (found < 0) continue
      const mine = counts[index]
      const other = otherCounts[found]
      total += multiply ? mine * other : mine < other ? mine : other
      low = found + 1
    }
    return total
  }
  let index = 0
  let otherIndex = 0
  while (index < keys.length && otherIndex < otherKeys.length) {
    if (bounded && total + remaining[index] < minimumShared) return total
    const key = keys[index]
    const otherKey = otherKeys[otherIndex]
    if (key === otherKey) {
      const mine = counts[index]
      const other = otherCounts[otherIndex]
      total += multiply ? mine * other : mine < other ? mine : other
      index++
      otherIndex++
      continue
    }
    if (key < otherKey) index++
    else otherIndex++
  }
  return total
}

/**
 * Whichever pair of representations turned up, reduced to the four cases that
 * exist. Orientation is normalized here rather than written twice: both
 * operations are symmetric, so a trie-first mixed pair is the packed-first one
 * with its operands swapped.
 *
 * A domain mismatch is `0` without a walk — `'a'` and `97` are different
 * elements at this layer, so two profiles that packed different domains share
 * no gram by definition.
 */
function combine(a: NGramProfile, b: NGramProfile, multiply: boolean): number {
  // Two depths share no gram, and packing is where that stops being structural:
  // a trie of depth 1 could never line up with one of depth 2, while the
  // unigram `[97]` and the bigram `[0, 97]` both key to 97. A scorer compares
  // profiles it prepared itself and cannot reach this, which is why the
  // prepared kernels below do not repeat the test per candidate.
  if (a.gramSize !== b.gramSize) return 0
  const storageA = a.storage
  const storageB = b.storage
  if (storageA.kind === 'packed') {
    if (storageB.kind === 'trie') {
      return packedAgainstTrie(storageA, storageB.root, a.gramSize, multiply)
    }
    return storageA.elementDomain === storageB.elementDomain
      ? packedIntersect(storageA, storageB, multiply, null, 0)
      : 0
  }
  if (storageB.kind === 'packed') {
    return packedAgainstTrie(storageB, storageA.root, a.gramSize, multiply)
  }
  return multiply
    ? dotProductTries(storageA.root, storageB.root, a.gramSize)
    : sharedFrequencyTries(storageA.root, storageB.root, a.gramSize)
}

/** `Σ min(a_g, b_g)` over the grams the two profiles share. */
export function sharedFrequency(a: NGramProfile, b: NGramProfile): number {
  return combine(a, b, false)
}

/** `Σ a_g · b_g` over the grams the two profiles share. */
export function dotProduct(a: NGramProfile, b: NGramProfile): number {
  return combine(a, b, true)
}

/**
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
function sharedFrequencyTries(
  rootA: GramNode,
  rootB: GramNode,
  gramSize: number,
): number {
  if (gramSize === 1) return sharedCounts(rootA, rootB)
  if (gramSize === 2) {
    const childrenB = childrenOf(rootB)
    let shared = 0
    for (const [element, childA] of childrenOf(rootA)) {
      const childB = childrenB.get(element)
      if (childB !== undefined) shared += sharedCounts(childA, childB)
    }
    return shared
  }
  if (gramSize === 3) {
    const childrenB = childrenOf(rootB)
    let shared = 0
    for (const [first, childA] of childrenOf(rootA)) {
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
  const nodesA: GramNode[] = [rootA]
  const nodesB: GramNode[] = [rootB]
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
 * A second literal traversal rather than {@link sharedFrequencyTries} taking a
 * combiner: the innermost frame of a walk is the last place to put a callback
 * every n-gram metric would make megamorphic.
 */
function dotProductTries(rootA: GramNode, rootB: GramNode, gramSize: number): number {
  if (gramSize === 1) return dotCounts(rootA, rootB)
  if (gramSize === 2) {
    const childrenB = childrenOf(rootB)
    let product = 0
    for (const [element, childA] of childrenOf(rootA)) {
      const childB = childrenB.get(element)
      if (childB !== undefined) product += dotCounts(childA, childB)
    }
    return product
  }
  if (gramSize === 3) {
    const childrenB = childrenOf(rootB)
    let product = 0
    for (const [first, childA] of childrenOf(rootA)) {
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
  const nodesA: GramNode[] = [rootA]
  const nodesB: GramNode[] = [rootB]
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

/** A query's trie side compiled once, then run against each candidate's root. */
interface TrieWalk {
  (choiceRoot: GramNode): number
}

/** {@link TrieWalk} that may stop short of the true count — see `BoundedFrequencyKernel`. */
interface BoundedTrieWalk {
  (choiceRoot: GramNode, minimumShared: number): number
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
function remainingTotals(totals: ArrayLike<number>): Uint32Array {
  const remaining = new Uint32Array(totals.length + 1)
  for (let index = totals.length - 1; index >= 0; index--) {
    remaining[index] = remaining[index + 1] + totals[index]
  }
  return remaining
}

function flattenUnigrams(root: GramNode): [unknown[], number[], Uint32Array] {
  const keys: unknown[] = []
  const frequencies: number[] = []
  for (const [element, count] of countsOf(root)) {
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

function flattenBigrams(root: GramNode): FlatLevel {
  const firstKeys: unknown[] = []
  const secondKeys: unknown[][] = []
  const frequencies: number[][] = []
  const totals: number[] = []
  for (const [first, child] of childrenOf(root)) {
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

function flattenTrigrams(root: GramNode): FlatTrigramLevel {
  const firstKeys: unknown[] = []
  const secondKeys: unknown[][] = []
  const thirdKeys: unknown[][][] = []
  const frequencies: number[][][] = []
  const totals: number[] = []
  for (const [first, child] of childrenOf(root)) {
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
 * Every gram a trie holds, with its frequency. Iterative over an explicit
 * stack for the reason the walks are: `gramSize` is the trie's depth and comes
 * from the caller.
 */
function eachGram(
  root: GramNode,
  gramSize: number,
  visit: (elements: readonly unknown[], count: number) => void,
): void {
  const last = gramSize - 1
  const nodes: GramNode[] = [root]
  const prefixes: unknown[][] = [[]]
  let top = 1
  while (top > 0) {
    top--
    const node = nodes[top]
    const prefix = prefixes[top]
    if (prefix.length === last) {
      for (const [element, count] of countsOf(node)) visit([...prefix, element], count)
      continue
    }
    for (const [element, child] of childrenOf(node)) {
      nodes[top] = child
      prefixes[top] = [...prefix, element]
      top++
    }
  }
}

/**
 * A trie spelling out what a packed query holds, built once so that a run of
 * trie candidates costs the flattened walk below and no decoding at all.
 */
function trieFromPacked(packed: PackedGrams, gramSize: number): GramNode {
  const root = emptyNode()
  const digits = new Array<number>(gramSize)
  const last = gramSize - 1
  for (let index = 0; index < packed.keys.length; index++) {
    unpackGram(packed.keys[index], gramSize, packed.radix, digits)
    let node = root
    for (let offset = 0; offset < last; offset++) {
      let children = node.children
      if (children === null) {
        children = new Map<unknown, GramNode>()
        node.children = children
      }
      const element = domainElement(digits[offset], packed.elementDomain)
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
    counts.set(domainElement(digits[last], packed.elementDomain), packed.counts[index])
  }
  return root
}

/** A packed query side, with the suffix totals its bound reads. */
interface PackedProjection {
  readonly grams: PackedGrams
  readonly remaining: Uint32Array
}

/**
 * A trie query re-spelled as packed keys, built once against the first packed
 * candidate that turns up.
 *
 * Only the grams this domain and radix can hold: a packed candidate proves
 * every one of *its* gram elements fits, so a query gram that does not fit can
 * never match one and dropping it changes no answer. That is also why
 * `remaining` is summed from the projected frequencies — the strongest bound
 * that stays correct, where the full query's totals would merely prune later.
 *
 * The radix comes from the candidate rather than from `gramSize`, so the two
 * sides are spelled at the same scale by construction. Every packed profile of
 * a given depth uses the one canonical radix, so a later candidate matches it.
 */
function packedProjection(
  root: GramNode,
  gramSize: number,
  domain: ElementDomain,
  radix: number,
): PackedProjection {
  const found: Array<{ key: number; count: number }> = []
  eachGram(root, gramSize, (elements, count) => {
    let key = 0
    for (let offset = 0; offset < gramSize; offset++) {
      const digit = packingDigit(elements[offset], domain, radix)
      if (digit < 0) return
      key = key * radix + digit
    }
    found.push({ key, count })
  })
  found.sort((left, right) => left.key - right.key)
  const keys = new Float64Array(found.length)
  const counts = new Uint32Array(found.length)
  const totals = new Uint32Array(found.length)
  for (let index = 0; index < found.length; index++) {
    keys[index] = found[index].key
    counts[index] = found[index].count
    totals[index] = found[index].count
  }
  return {
    grams: { kind: 'packed', elementDomain: domain, radix, keys, counts },
    remaining: remainingTotals(totals),
  }
}

/**
 * The query's trie flattened once per depth, so a candidate costs one `Map.get`
 * per query gram or group rather than a walk of both sides.
 */
function trieSharedWalk(root: GramNode, gramSize: number): BoundedTrieWalk {
  if (gramSize === 1) {
    const [keys, frequencies, remaining] = flattenUnigrams(root)
    return (choiceRoot, minimumShared) => {
      const counts = countsOf(choiceRoot)
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
    const { firstKeys, secondKeys, frequencies, remaining } = flattenBigrams(root)
    return (choiceRoot, minimumShared) => {
      const children = childrenOf(choiceRoot)
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
      flattenTrigrams(root)
    return (choiceRoot, minimumShared) => {
      const children = childrenOf(choiceRoot)
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
  return (choiceRoot) => sharedFrequencyTries(root, choiceRoot, gramSize)
}

function trieDotWalk(root: GramNode, gramSize: number): TrieWalk {
  if (gramSize === 1) {
    const [keys, frequencies] = flattenUnigrams(root)
    return (choiceRoot) => {
      const counts = countsOf(choiceRoot)
      let product = 0
      for (let index = 0; index < keys.length; index++) {
        const other = counts.get(keys[index])
        if (other !== undefined) product += frequencies[index] * other
      }
      return product
    }
  }
  if (gramSize === 2) {
    const { firstKeys, secondKeys, frequencies } = flattenBigrams(root)
    return (choiceRoot) => {
      const children = childrenOf(choiceRoot)
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
    const { firstKeys, secondKeys, thirdKeys, frequencies } = flattenTrigrams(root)
    return (choiceRoot) => {
      const children = childrenOf(choiceRoot)
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
  return (choiceRoot) => dotProductTries(root, choiceRoot, gramSize)
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
 *
 * A candidate stored the other way round is answered through a query-side
 * representation compiled on first use and kept, and carries the same bound the
 * matching path does. What must never happen again for a later candidate of
 * that storage is the compiling: no decode, projection, sort or allocation
 * belonging to the query may sit inside the candidate loop.
 */
export function sharedFrequencyKernel(query: NGramProfile): BoundedFrequencyKernel {
  const storage = query.storage
  const gramSize = query.gramSize
  if (storage.kind === 'packed') {
    const remaining = remainingTotals(storage.counts)
    // Compiled when a differently stored candidate first turns up, and never
    // again: a decode per candidate is the one thing a prepared kernel may not
    // do. Deferred rather than eager because most corpora are all one storage,
    // and the query that never meets the other kind must not pay for it.
    let spelledOut: BoundedTrieWalk | null = null
    return (choice, minimumShared) => {
      const other = choice.storage
      if (other.kind === 'packed') {
        return other.elementDomain === storage.elementDomain
          ? packedIntersect(storage, other, false, remaining, minimumShared)
          : 0
      }
      spelledOut ??= trieSharedWalk(trieFromPacked(storage, gramSize), gramSize)
      return spelledOut(other.root, minimumShared)
    }
  }
  const walk = trieSharedWalk(storage.root, gramSize)
  // One projection per element domain, since a query trie may hold grams of
  // both and a candidate is packed in exactly one.
  let numbers: PackedProjection | null = null
  let chars: PackedProjection | null = null
  return (choice, minimumShared) => {
    const other = choice.storage
    if (other.kind === 'trie') return walk(other.root, minimumShared)
    if (other.elementDomain === 'number') {
      numbers ??= packedProjection(storage.root, gramSize, 'number', other.radix)
      return packedIntersect(
        numbers.grams,
        other,
        false,
        numbers.remaining,
        minimumShared,
      )
    }
    chars ??= packedProjection(storage.root, gramSize, 'char', other.radix)
    return packedIntersect(chars.grams, other, false, chars.remaining, minimumShared)
  }
}

export function dotProductKernel(query: NGramProfile): FrequencyKernel {
  const storage = query.storage
  const gramSize = query.gramSize
  if (storage.kind === 'packed') {
    let spelledOut: TrieWalk | null = null
    return (choice) => {
      const other = choice.storage
      if (other.kind === 'packed') {
        return other.elementDomain === storage.elementDomain
          ? packedIntersect(storage, other, true, null, 0)
          : 0
      }
      spelledOut ??= trieDotWalk(trieFromPacked(storage, gramSize), gramSize)
      return spelledOut(other.root)
    }
  }
  const walk = trieDotWalk(storage.root, gramSize)
  let numbers: PackedProjection | null = null
  let chars: PackedProjection | null = null
  return (choice) => {
    const other = choice.storage
    if (other.kind === 'trie') return walk(other.root)
    if (other.elementDomain === 'number') {
      numbers ??= packedProjection(storage.root, gramSize, 'number', other.radix)
      return packedIntersect(numbers.grams, other, true, null, 0)
    }
    chars ??= packedProjection(storage.root, gramSize, 'char', other.radix)
    return packedIntersect(chars.grams, other, true, null, 0)
  }
}
