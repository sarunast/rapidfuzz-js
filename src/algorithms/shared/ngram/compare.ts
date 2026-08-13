import { canonicalRadix, unpackGram } from './key.js'
import { domainElement, domainOf, packedKeys } from './packing.js'
import {
  childrenOf,
  countsOf,
  gramsIn,
  type GramNode,
  type NGramProfile,
  type PackedGrams,
} from './profile.js'

/**
 * `Σ min(a_g, b_g)` for one comparison, without building a profile of either
 * side, or `null` where the pair cannot be packed and the profiles have to
 * answer instead.
 *
 * A prepared profile sorts its grams so that later queries can merge against
 * it. A direct comparison has no later query, so it is paying `O(n log n)` to
 * order something it reads once: this tallies the smaller side into a `Map` and
 * spends the larger against those counts, which is `O(n + m)`. Decrementing as
 * it goes is what makes one map enough — a gram runs out exactly when the
 * smaller side's count of it does, so the walk never needs the larger side's
 * own frequencies. Cosine gets no such shortcut, because `Σ b_g²` needs them.
 *
 * Answers any depth with a canonical radix, and is worth calling on far fewer:
 * Dice's direct path owns that policy and states what was measured.
 */
export function directSharedFrequency(
  a: ArrayLike<unknown>,
  b: ArrayLike<unknown>,
  gramSize: number,
): number | null {
  const radix = canonicalRadix(gramSize)
  if (radix === null) return null
  const gramsA = gramsIn(a, gramSize)
  const gramsB = gramsIn(b, gramSize)
  if (gramsA === 0 || gramsB === 0) return null
  const domain = domainOf(a)
  // Two first elements in different domains cannot share one packed key space,
  // where `'b'` and `98` would become the same key. They may still share later
  // grams — `['x', 1, 2]` and `[9, 1, 2]` both hold `[1, 2]` — so this defers to
  // the profile path, which answers such a pair exactly. Never `0`.
  if (domainOf(b) !== domain) return null

  // The smaller side is the one tallied, so the `Map` stays as small as the
  // pair allows and the larger side's walk is lookups rather than insertions.
  const aFirst = gramsA <= gramsB
  const smallCount = aFirst ? gramsA : gramsB
  const largeCount = aFirst ? gramsB : gramsA
  const smallKeys = packedKeys(aFirst ? a : b, gramSize, smallCount, radix, domain)
  if (smallKeys === null) return null
  const largeKeys = packedKeys(aFirst ? b : a, gramSize, largeCount, radix, domain)
  if (largeKeys === null) return null

  const counts = new Map<number, number>()
  for (let index = 0; index < smallCount; index++) {
    const key = smallKeys[index]
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  let shared = 0
  for (let index = 0; index < largeCount; index++) {
    const key = largeKeys[index]
    const remaining = counts.get(key)
    if (remaining === undefined || remaining === 0) continue
    counts.set(key, remaining - 1)
    // Every gram *occurrence* the smaller side had has now been consumed, and
    // `Σ min` cannot exceed that count, so nothing left on the larger side can
    // change the answer. Exact, not a cutoff — the same number the full walk
    // returns. Both key arrays were built, and so validated, before this loop,
    // so stopping early cannot answer for a pair that should have declined.
    // Worth 2.3-3.7x on public Dice where it fires and nothing where it does
    // not, which `bench/ngram.bench.ts` holds at 0.99-1.05x.
    if (++shared === smallCount) return shared
  }
  return shared
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
 * A packed side against a trie one: every packed gram decoded back to its
 * elements and looked up.
 *
 * One function for both operations, and one orientation for both operand
 * orders, because `Σ min` and `Σ a·b` are each symmetric. The trie walks below
 * are literal per depth on a recorded measurement; this path is the rare one and
 * gets no such licence. It needs a sequence packing refused, which narrows with
 * depth as the canonical radix does: an astral code point at trigram depth, or
 * anything outside Latin-1 at depths 4 to 6, where the widest feasible rung is
 * `0x100`.
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
export function packedIntersect(
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
  // profiles it prepared itself and cannot reach this, which is why
  // `sharedFrequencyKernel` and `dotProductKernel` do not repeat the test per
  // candidate.
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
 * Depths 1-3 get literal loops; any deeper `gramSize` — the option accepts every
 * safe integer from 1 up — takes the generic walk. The generic walk
 * allocates three stack arrays per comparison, which measured 1.6x the
 * specialized bigram loop over 100 queries against 1000 prepared choices — and
 * over prebuilt trigram profiles, at four lengths from 12 to 512 characters,
 * 1.2-1.7x the literal trigram loop below.
 *
 * Deeper than that it is iterative over an explicit stack, not recursive:
 * `gramSize` is caller-supplied and equals the trie depth, so recursion would
 * put a stack overflow inside the range of valid inputs.
 */
export function sharedFrequencyTries(
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
export function dotProductTries(
  rootA: GramNode,
  rootB: GramNode,
  gramSize: number,
): number {
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
