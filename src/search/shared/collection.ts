import type { Items } from '../types.js'

export interface SourceEntry<TItem, TKey = unknown> {
  readonly item: TItem
  readonly key: TKey
}

export function assertCollection(value: unknown): void {
  if (typeof value === 'string') {
    throw new TypeError('items must be a collection, not a single string')
  }
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('items must be an array, iterable, map, or plain object')
  }
  if (isMapLike(value) || isIterable(value)) return
  if (!isPlainRecord(value)) {
    throw new TypeError('items must be an array, iterable, map, or plain object')
  }
}

function isPlainRecord(value: object): boolean {
  const prototype = Object.getPrototypeOf(value)
  return prototype === null || Object.getPrototypeOf(prototype) === null
}

function isMapLike(value: object): value is ReadonlyMap<unknown, unknown> {
  return (
    typeof Reflect.get(value, 'get') === 'function' &&
    typeof Reflect.get(value, 'has') === 'function' &&
    typeof Reflect.get(value, 'entries') === 'function' &&
    typeof Reflect.get(value, Symbol.iterator) === 'function'
  )
}

function isIterable(value: object): value is Iterable<unknown> {
  return typeof Reflect.get(value, Symbol.iterator) === 'function'
}

export function* collectionEntries<TItem>(
  items: Items<TItem>,
): Generator<SourceEntry<TItem>> {
  assertCollection(items)
  if (isMapLike(items)) {
    for (const [key, item] of items) yield { item, key }
    return
  }
  if (isIterable(items)) {
    let key = 0
    for (const item of items) yield { item, key: key++ }
    return
  }
  for (const key of Object.keys(items)) yield { item: items[key], key }
}
