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
  if (domainOf(b) !== domain) return null

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

const PROBE_LENGTH_RATIO = 8

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

function combine(a: NGramProfile, b: NGramProfile, multiply: boolean): number {
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

export function sharedFrequency(a: NGramProfile, b: NGramProfile): number {
  return combine(a, b, false)
}

export function dotProduct(a: NGramProfile, b: NGramProfile): number {
  return combine(a, b, true)
}

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
