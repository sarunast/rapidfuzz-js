import type { Match } from '../results.js'
import { collectionEntries } from '../shared/collection.js'
import type { Items } from '../types.js'

/**
 * Which source item and key each dense choice id stands for.
 *
 * Flat and parallel rather than one row object per choice, and the whole of
 * what the two Matchers share: an id names a choice, and how that choice is
 * scored is the constructor's business rather than the table's.
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
 * Establishes dense choice identity, and nothing else.
 *
 * `accept` is called for every source item in collection order, with the id
 * that item **would** hold — a candidate, which becomes the item's id only when
 * the call answers `true`. A `false` consumes no id, so the next item is
 * offered the same one.
 *
 * Whatever the caller wants beside that id has to be built or stored **inside
 * the call**: a reader may borrow rather than copy, handing back one mutable
 * buffer every time, so a value collected here and consumed after the walk can
 * be the next choice's by then. That lifetime is the reason this takes a
 * callback rather than returning the values it read.
 *
 * A table knows an item, its key and its id. What sits beside the id — a
 * prepared handle, a posting list — it never learns.
 */
export function buildChoiceTable<TItem>(
  items: Items<TItem>,
  accept: (item: TItem, id: number) => boolean,
): ChoiceTable<TItem> {
  let keys: unknown[] | null = null
  let position = 0
  // Two literal loops, so the common walk reads its key off the counter rather
  // than off an entry object it never had to allocate, and can size what it
  // fills from what it is reading. That sizing is what pays for a second array
  // per choice: over 2,000 choices, growing by push measured 31.8ns a choice
  // against 29.0 for the row objects this layout replaces, and sized once 26.0.
  // Reads cost the same either way, 1.28ns, against an order control.
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
