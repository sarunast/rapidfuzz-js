import type { IndexedMatcherWorkload } from './shared.ts'

export const ARBITRARY_QUERY_ELEMENTS = 100_001

class ArbitraryElement {
  toString(): string {
    return 'arbitrary-element-sentinel'
  }
}

class ArbitraryQuery {
  [index: number]: unknown
  readonly length = ARBITRARY_QUERY_ELEMENTS

  constructor(sentinel: ArbitraryElement) {
    for (let index = 0; index < this.length - 1; index++) this[index] = index
    this[this.length - 1] = sentinel
  }
}

/**
 * The rare path a direct index takes when a query holds something no integer
 * key can spell: the whole query is ordinalized locally to count its own grams,
 * which is the one query shape that allocates a table proportional to it. The
 * temporary wrapper, the sentinel and that table all die on return.
 */
export function runArbitraryElementQuery(matcher: IndexedMatcherWorkload): void {
  matcher.best(new ArbitraryQuery(new ArbitraryElement()))
}
