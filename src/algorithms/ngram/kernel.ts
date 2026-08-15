import { dotProductTries, packedIntersect, sharedFrequencyTries } from './compare.js'
import { unpackGram } from './key.js'
import { domainElement, packingDigit, type ElementDomain } from './packing.js'
import {
  childrenOf,
  countsOf,
  emptyNode,
  type GramNode,
  type NGramProfile,
  type PackedGrams,
} from './profile.js'

export interface FrequencyKernel {
  (choice: NGramProfile): number
}

export interface BoundedFrequencyKernel {
  (choice: NGramProfile, minimumShared: number): number
}

interface TrieWalk {
  (choiceRoot: GramNode): number
}

interface BoundedTrieWalk {
  (choiceRoot: GramNode, minimumShared: number): number
}

interface FlatBigramLevel {
  readonly firstKeys: unknown[]
  readonly secondKeys: unknown[][]
  readonly frequencies: number[][]
  readonly remaining: Uint32Array
}

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

function flattenBigrams(root: GramNode): FlatBigramLevel {
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

interface PackedProjection {
  readonly grams: PackedGrams
  readonly remaining: Uint32Array
}

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
  for (let index = 0; index < found.length; index++) {
    keys[index] = found[index].key
    counts[index] = found[index].count
  }
  return {
    grams: { kind: 'packed', elementDomain: domain, radix, keys, counts },
    remaining: remainingTotals(counts),
  }
}

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

export function sharedFrequencyKernel(query: NGramProfile): BoundedFrequencyKernel {
  const storage = query.storage
  const gramSize = query.gramSize
  if (storage.kind === 'packed') {
    const remaining = remainingTotals(storage.counts)
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
