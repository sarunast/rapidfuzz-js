import type { Match } from '../results.js'
import { collectionEntries } from '../shared/collection.js'
import type { Items } from '../types.js'

export interface ChoiceTable<TItem> {
  readonly items: readonly TItem[]
  readonly keys: readonly unknown[] | null
}

export function buildChoiceTable<TItem>(
  items: Items<TItem>,
  accept: (item: TItem, id: number) => boolean,
): ChoiceTable<TItem> {
  let keys: unknown[] | null = null
  let position = 0
  if (Array.isArray(items)) {
    const count = items.length
    const kept: TItem[] = new Array(count)
    for (let key = 0; key < count; key++) {
      const item = items[key]
      if (!accept(item, position)) continue
      if (keys === null && key !== position) {
        keys = []
        for (let index = 0; index < position; index++) keys.push(index)
      }
      if (keys !== null) keys.push(key)
      kept[position] = item
      position++
    }
    kept.length = position
    return { items: kept, keys }
  }
  const kept: TItem[] = []
  for (const entry of collectionEntries(items)) {
    if (!accept(entry.item, position)) continue
    const key = entry.key
    if (keys === null && key !== position) {
      keys = []
      for (let index = 0; index < position; index++) keys.push(index)
    }
    if (keys !== null) keys.push(key)
    kept.push(entry.item)
    position++
  }
  return { items: kept, keys }
}

export function keyAt<TItem>(table: ChoiceTable<TItem>, id: number): unknown {
  const keys = table.keys
  return keys === null ? id : keys[id]
}

export function matchAt<TItem>(
  table: ChoiceTable<TItem>,
  id: number,
  score: number,
): Match<TItem, unknown> {
  return { item: table.items[id], key: keyAt(table, id), score }
}
