import type { OptimumProof } from '#core/scoring/optimumProof.js'
import type { Sequence } from '#core/types.js'

import { prepareTokenChoice, preparedTokenChoice, uniqueOf } from './tokens.js'

const SUBSET_CAP = 12

function setKeyOf(sorted: readonly string[]): string {
  let key = ''
  for (let i = 0; i < sorted.length; i++) key += `${sorted[i].length}:${sorted[i]}`
  return key
}

interface ContainmentIndex {
  readonly postings: Map<string, number[]>
  readonly bySet: Map<string, number[]>
  readonly keysById: readonly (ReadonlySet<string> | null)[]
}

function build(prepared: readonly unknown[]): ContainmentIndex {
  const postings = new Map<string, number[]>()
  const bySet = new Map<string, number[]>()
  const keysById: (ReadonlySet<string> | null)[] = new Array(prepared.length)

  for (let id = 0; id < prepared.length; id++) {
    const unique = uniqueOf(preparedTokenChoice(prepared[id]))
    const packed = [...unique.packed.keys()]
    if (packed.length === 0) {
      keysById[id] = null
      continue
    }
    keysById[id] = new Set(packed)
    for (const key of packed) {
      let list = postings.get(key)
      if (list === undefined) postings.set(key, (list = []))
      list.push(id)
    }
    if (unique.mixed.size === 0) {
      const setKey = setKeyOf(packed.slice().sort())
      let ids = bySet.get(setKey)
      if (ids === undefined) bySet.set(setKey, (ids = []))
      ids.push(id)
    }
  }

  return { postings, bySet, keysById }
}

function queryKeys(query: Sequence): string[] | undefined {
  const unique = uniqueOf(prepareTokenChoice(query))
  if (unique.mixed.size !== 0) return undefined

  const keys = [...unique.packed.keys()]
  if (keys.length === 0 || keys.length > SUBSET_CAP) return undefined
  return keys
}

function rarestPosting(
  index: ContainmentIndex,
  keys: readonly string[],
): number[] | undefined {
  let rarest: number[] | undefined
  for (let i = 0; i < keys.length; i++) {
    const list = index.postings.get(keys[i])
    if (list === undefined) return undefined
    if (rarest === undefined || list.length < rarest.length) rarest = list
  }
  return rarest
}

function holdsEvery(choiceKeys: ReadonlySet<string>, keys: readonly string[]): boolean {
  for (let i = 0; i < keys.length; i++) {
    if (!choiceKeys.has(keys[i])) return false
  }
  return true
}

class SmallestIds {
  private readonly ids: number[] = []
  private readonly seen = new Set<number>()

  constructor(private readonly limit: number) {}

  add(id: number): void {
    if (this.seen.has(id)) return
    if (this.ids.length === this.limit && id > this.ids[this.ids.length - 1]) return

    let at = this.ids.length
    while (at > 0 && this.ids[at - 1] > id) at--
    this.ids.splice(at, 0, id)
    this.seen.add(id)

    if (this.ids.length > this.limit) {
      const dropped = this.ids[this.ids.length - 1]
      this.ids.length = this.limit
      this.seen.delete(dropped)
    }
  }

  get full(): boolean {
    return this.ids.length === this.limit
  }

  get collected(): readonly number[] {
    return this.ids
  }
}

function offerSubsets(
  index: ContainmentIndex,
  sorted: readonly string[],
  wanted: number,
  into: (id: number) => void,
): void {
  const total = 1 << sorted.length
  for (let mask = 1; mask < total; mask++) {
    const subset: string[] = []
    for (let bit = 0; bit < sorted.length; bit++) {
      if ((mask & (1 << bit)) !== 0) subset.push(sorted[bit])
    }
    const ids = index.bySet.get(setKeyOf(subset))
    if (ids === undefined) continue
    for (let i = 0; i < ids.length && i < wanted; i++) into(ids[i])
  }
}

export function tokenContainmentProof(prepared: readonly unknown[]): OptimumProof {
  let index: ContainmentIndex | null = null

  return {
    best(query: Sequence): number | undefined {
      const keys = queryKeys(query)
      if (keys === undefined) return undefined
      const built = (index ??= build(prepared))

      let earliest = -1
      const rarest = rarestPosting(built, keys)
      if (rarest !== undefined) {
        for (let i = 0; i < rarest.length; i++) {
          const id = rarest[i]
          const choiceKeys = built.keysById[id]
          if (choiceKeys !== null && holdsEvery(choiceKeys, keys)) {
            earliest = id
            break
          }
        }
      }

      const sorted = keys.slice().sort()
      offerSubsets(built, sorted, 1, (id) => {
        if (earliest === -1 || id < earliest) earliest = id
      })

      return earliest === -1 ? undefined : earliest
    },

    top(query: Sequence, limit: number): readonly number[] | undefined {
      const keys = queryKeys(query)
      if (keys === undefined) return undefined
      const built = (index ??= build(prepared))

      const smallest = new SmallestIds(limit)
      const rarest = rarestPosting(built, keys)
      if (rarest !== undefined) {
        for (let i = 0; i < rarest.length && !smallest.full; i++) {
          const id = rarest[i]
          const choiceKeys = built.keysById[id]
          if (choiceKeys !== null && holdsEvery(choiceKeys, keys)) smallest.add(id)
        }
      }

      const sorted = keys.slice().sort()
      offerSubsets(built, sorted, limit, (id) => {
        smallest.add(id)
      })

      return smallest.full ? smallest.collected : undefined
    },
  }
}
