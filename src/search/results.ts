/** One search result: what matched, where it lives, and how well it scored. */
export interface Match<TItem, TKey> {
  /** Your original item, never the text extracted or normalized from it. */
  readonly item: TItem
  /**
   * Where the item sits in the collection you passed — an array index, a `Map`
   * key, an object property name, or the position an iterable yielded it at.
   *
   * Note that filtering a collection through a generator renumbers this: the
   * position is counted over what was yielded, not over the original source.
   */
  readonly key: TKey
  /** On the scorer's own scale. */
  readonly score: number
}
