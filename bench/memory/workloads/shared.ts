export interface IndexedMatcherWorkload {
  readonly size: number
  best(query: ArrayLike<unknown> | string): unknown
  search(
    query: ArrayLike<unknown> | string,
    options?: { readonly limit?: number | null },
  ): readonly unknown[]
}

const LOWER = 'abcdefghijklmnopqrstuvwxyz'

export function lowercaseBigramCorpus(count: number): string[] {
  const corpus = new Array<string>(count)
  for (let index = 0; index < count; index++) {
    const cycle = index % (LOWER.length * LOWER.length)
    corpus[index] = LOWER[Math.floor(cycle / LOWER.length)] + LOWER[cycle % LOWER.length]
  }
  return corpus
}

function mix(value: number): number {
  let mixed = value >>> 0
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x7feb_352d)
  mixed = Math.imul(mixed ^ (mixed >>> 15), 0x846c_a68b)
  return (mixed ^ (mixed >>> 16)) >>> 0
}

/** A deterministic query derived directly from its operation number. */
export function ordinaryQuery(operation: number): string {
  let state = mix(operation + 0x9e37_79b9)
  let query = ''
  for (let index = 0; index < 24; index++) {
    state = mix(state + index + operation)
    query += LOWER[state % LOWER.length]
  }
  // The suffix is an injective base-26 encoding over the soak's operation range.
  let ordinal = operation
  for (let index = 0; index < 6; index++) {
    query += LOWER[ordinal % LOWER.length]
    ordinal = Math.floor(ordinal / LOWER.length)
  }
  return query
}

export function unrelatedQuery(operation: number): string {
  return ordinaryQuery(operation).toUpperCase()
}

export function runOrdinaryBest(
  matcher: IndexedMatcherWorkload,
  operation: number,
): void {
  matcher.best(ordinaryQuery(operation))
}
