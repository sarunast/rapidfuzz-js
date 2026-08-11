import type { Items } from './types.js'

export interface SourceEntry<T, K = unknown> {
  readonly item: T
  readonly key: K
}

function assertCollection(value: unknown): void {
  if (typeof value === 'string') {
    throw new TypeError('items must be a collection, not a single string')
  }
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('items must be an array, iterable, map, or object')
  }
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

export function* collectionEntries<T>(items: Items<T>): Generator<SourceEntry<T>> {
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
  for (const [key, item] of Object.entries(items)) yield { item, key }
}
