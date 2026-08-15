import type { Direction, Normalizer } from '../../core/types.js'
import { collectionEntries } from '../shared/collection.js'
import type { SourceEntry } from '../shared/collection.js'
import type { ChoiceReader, ReaderOptions } from '../shared/readers.js'
import type { AnyMatcherOptions, Items } from '../types.js'

export function stableOptionsOf<TItem, TDirection extends Direction, TBrand>(
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
  normalize: Normalizer | undefined,
): ReaderOptions<TItem, TBrand> {
  return {
    getText: options.getText,
    getPrepared: options.getPrepared,
    normalize,
    missingItems: options.missingItems,
  }
}

export function better(direction: Direction, score: number, current: number): boolean {
  return direction === 'similarity' ? score > current : score < current
}

export function arrayItemsOf<TItem>(items: Items<TItem>): readonly TItem[] | null {
  return Array.isArray(items) ? items : null
}

export function* presentEntries<TItem>(
  items: Items<TItem>,
  choices: ChoiceReader<TItem>,
): Generator<SourceEntry<TItem>> {
  const arrayItems = arrayItemsOf(items)
  if (arrayItems !== null) {
    for (let key = 0; key < arrayItems.length; key++) {
      const item = arrayItems[key]
      if (choices.present(item)) yield { item, key }
    }
    return
  }
  for (const entry of collectionEntries(items)) {
    if (choices.present(entry.item)) yield entry
  }
}
