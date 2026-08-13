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
  // An object that is neither is read as a record of keys to items, and only a
  // plain one carries items that way. A `Date` or a `Promise` keeps its state
  // somewhere Object.keys cannot see, so accepting it would answer a wrong
  // argument with an empty collection instead of an error.
  if (!isPlainRecord(value)) {
    throw new TypeError('items must be an array, iterable, map, or plain object')
  }
}

// A plain object's prototype is `Object.prototype`, whose own prototype is
// null; comparing the depth rather than the identity accepts one built in
// another realm, where `Object.prototype` is a different object.
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
  // Object.keys over Object.entries: the same walk without a two-element
  // array allocated for every property.
  for (const key of Object.keys(items)) yield { item: items[key], key }
}
