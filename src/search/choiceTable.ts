import { collectionEntries } from './collection.js'
import type { Match } from './results.js'
import type { Items } from './types.js'

/**
 * Which source item and key each dense choice id stands for.
 *
 * Flat and parallel rather than one row object per choice, and the shape both
 * Matchers agree on: an id indexes `items` here, the scoring representation
 * beside it, and nothing else needs to know which of the two a search walked.
 */
export interface ChoiceTable<TItem> {
  readonly items: readonly TItem[]
  /**
   * `null` while every key is its own position, which is an array with no gaps —
   * the common case, and the difference between a second array and nothing.
   */
  readonly keys: readonly unknown[] | null
}

/**
 * A table and the values read alongside it, one per id.
 *
 * The values are handed back rather than kept on the table: what they are is
 * the difference between the two Matchers — one holds them as its prepared
 * array, the other feeds them to an index and drops them — and the table is the
 * half that is the same either way.
 */
export interface ChoiceTableBuild<TItem, TValue> {
  readonly table: ChoiceTable<TItem>
  readonly values: readonly TValue[]
}

/**
 * Reads a collection once, keeping the choices `read` answers for.
 *
 * `null` from `read` is the skip sentinel, and `values` holds exactly one entry
 * for every retained choice, in ascending id order. That contract is what makes
 * the table shareable: `values[id]` and `table.items[id]` are the same source
 * choice.
 */
export function buildChoiceTable<TItem, TValue>(
  items: Items<TItem>,
  read: (item: TItem) => TValue | null,
): ChoiceTableBuild<TItem, TValue> {
  let keys: unknown[] | null = null
  let position = 0
  // Two literal loops, for the same reason the drivers are two: an array knows
  // its own upper bound, and sizing both arrays once rather than growing them
  // is what pays for the second array. Measured over 2,000 choices: growing
  // 31.8ns a choice against 29.0 for the row objects this replaces, sized once
  // 26.0. Reads cost the same either way — 1.28ns, against an order control.
  if (Array.isArray(items)) {
    const count = items.length
    const kept: TItem[] = new Array(count)
    const values: TValue[] = new Array(count)
    for (let key = 0; key < count; key++) {
      const item = items[key]
      const value = read(item)
      if (value === null) continue
      if (keys === null && key !== position) {
        keys = []
        for (let index = 0; index < position; index++) keys.push(index)
      }
      if (keys !== null) keys.push(key)
      kept[position] = item
      values[position] = value
      position++
    }
    kept.length = position
    values.length = position
    return { table: { items: kept, keys }, values }
  }
  const kept: TItem[] = []
  const values: TValue[] = []
  for (const entry of collectionEntries(items)) {
    const value = read(entry.item)
    if (value === null) continue
    const key = entry.key
    if (keys === null && key !== position) {
      keys = []
      for (let index = 0; index < position; index++) keys.push(index)
    }
    if (keys !== null) keys.push(key)
    kept.push(entry.item)
    values.push(value)
    position++
  }
  return { table: { items: kept, keys }, values }
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
