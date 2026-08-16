import { convSequence, elementsEqual, isUnmatchableElement } from '../../core/sequence.js'
import type { Sequence } from '../../core/types.js'
import { canonicalRadix } from './key.js'
import { domainOf, packedKeys, type ElementDomain } from './packing.js'

export interface GramNode {
  children: Map<unknown, GramNode> | null
  counts: Map<unknown, number> | null
}

export interface PackedGrams {
  readonly kind: 'packed'
  readonly elementDomain: ElementDomain
  readonly radix: number
  readonly keys: Float64Array
  readonly counts: Uint32Array
}

interface GramTrie {
  readonly kind: 'trie'
  readonly root: GramNode
}

type ProfileStorage = PackedGrams | GramTrie

export class NGramProfile {
  constructor(
    readonly gramSize: number,
    readonly gramCount: number,
    readonly squaredNorm: number,
    readonly storage: ProfileStorage,
    readonly elements: ArrayLike<unknown> | null,
  ) {}
}

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

export function gramsIn(elements: ArrayLike<unknown>, gramSize: number): number {
  const gramCount = elements.length - gramSize + 1
  return gramCount > 0 ? gramCount : 0
}

function leafNode(): GramLeaf {
  return { children: null, counts: new Map<unknown, number>() }
}

function bigramProfile(elements: ArrayLike<unknown>, gramCount: number): NGramProfile {
  const children = new Map<unknown, GramLeaf>()
  const root: GramBranch = { children, counts: null }
  let first = elements[0]
  let lastUnmatchable = isUnmatchableElement(first) ? 0 : -1
  let squaredNorm = 0
  for (let start = 0; start < gramCount; start++) {
    const end = start + 1
    const second = elements[end]
    if (isUnmatchableElement(second)) lastUnmatchable = end
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
  let lastUnmatchable = isUnmatchableElement(second)
    ? 1
    : isUnmatchableElement(first)
      ? 0
      : -1
  let squaredNorm = 0
  for (let start = 0; start < gramCount; start++) {
    const end = start + 2
    const third = elements[end]
    if (isUnmatchableElement(third)) lastUnmatchable = end
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

export function packedProfile(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile | null {
  const gramCount = gramsIn(elements, gramSize)
  if (gramCount === 0) return null
  const radix = canonicalRadix(gramSize)
  if (radix === null) return null
  const domain = domainOf(elements)

  const sorted = packedKeys(elements, gramSize, gramCount, radix, domain)
  if (sorted === null) return null
  sorted.sort()

  let distinct = 0
  for (let index = 0; index < gramCount; index++) {
    if (index === 0 || sorted[index] !== sorted[index - 1]) distinct++
  }
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

export function trieProfile(
  elements: ArrayLike<unknown>,
  gramSize: number,
): NGramProfile {
  const gramCount = gramsIn(elements, gramSize)
  if (gramCount === 0) {
    return new NGramProfile(gramSize, 0, 0, trieStorage(emptyNode()), elements)
  }
  if (gramSize === 2) return bigramProfile(elements, gramCount)
  if (gramSize === 3) return trigramProfile(elements, gramCount)

  const root = emptyNode()
  let lastUnmatchable = -1
  for (let index = 0; index < gramSize - 1; index++) {
    if (isUnmatchableElement(elements[index])) lastUnmatchable = index
  }
  let squaredNorm = 0
  const last = gramSize - 1
  for (let start = 0; start < gramCount; start++) {
    const end = start + last
    if (isUnmatchableElement(elements[end])) lastUnmatchable = end
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

const NO_CHILDREN: ReadonlyMap<unknown, GramNode> = /* @__PURE__ */ new Map()
const NO_COUNTS: ReadonlyMap<unknown, number> = /* @__PURE__ */ new Map()

export function childrenOf(node: GramNode): ReadonlyMap<unknown, GramNode> {
  return node.children ?? NO_CHILDREN
}

export function countsOf(node: GramNode): ReadonlyMap<unknown, number> {
  return node.counts ?? NO_COUNTS
}

export function zeroGramSimilarity(a: NGramProfile, b: NGramProfile): number {
  const elementsA = a.elements
  const elementsB = b.elements
  return elementsA !== null && elementsB !== null && elementsEqual(elementsA, elementsB)
    ? 1
    : 0
}
