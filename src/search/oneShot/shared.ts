import type { Direction, Normalizer } from '../../core/types.js'
import { collectionEntries } from '../shared/collection.js'
import type { SourceEntry } from '../shared/collection.js'
import type { ChoiceReader } from '../shared/readers.js'
import type { AnyMatcherOptions, Items, ResolvedMatcherOptions } from '../types.js'

/**
 * Every option read exactly once, as `createMatcher` does it.
 *
 * The reader and the query have to be handed the same normalizer: an accessor
 * that answers one function to each passes the prepared-choice check and then
 * scores against a query normalized some other way, which is the silent
 * mismatch that check exists to refuse.
 */
export function stableOptionsOf<TItem, TDirection extends Direction, TBrand>(
  options: AnyMatcherOptions<TItem, TDirection, TBrand>,
  scorer: ResolvedMatcherOptions<TItem, Direction, TBrand>['scorer'],
  normalize: Normalizer | undefined,
): ResolvedMatcherOptions<TItem, Direction, TBrand> {
  return {
    scorer,
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

/**
 * Every choice that has text to score, in collection order.
 *
 * The walk a query with no text of its own takes: nothing is scored, so what a
 * result needs is only whether the choice was there. Cold by construction —
 * every caller of this has already settled that the query is empty.
 */
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
