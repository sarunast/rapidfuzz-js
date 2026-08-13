// A hand-written multiset oracle for the n-gram tests: deliberately the
// slowest correct thing, so a differential failure names the implementation
// rather than a second clever version of it.
//
// A module rather than a copy per file because three describes across two of
// them check against it — `packed storage` and `the two storages answer alike`
// in `representation.test.ts`, and `properties` in `compare.test.ts`.

/**
 * A profile the trie is checked against: one bucket per gram, keyed by a
 * position-aware structure rather than a serialized string, and using the
 * library's `===` so a gram holding `NaN` matches nothing — including itself.
 */
export function referenceGrams(
  elements: readonly unknown[],
  gramSize: number,
): Array<{ readonly gram: readonly unknown[]; count: number }> {
  const buckets: Array<{ readonly gram: readonly unknown[]; count: number }> = []
  for (let start = 0; start + gramSize <= elements.length; start++) {
    const gram = elements.slice(start, start + gramSize)
    if (gram.some((element) => typeof element === 'number' && Number.isNaN(element))) {
      continue
    }
    const existing = buckets.find(
      (bucket) =>
        bucket.gram.length === gram.length &&
        bucket.gram.every((element, index) => element === gram[index]),
    )
    if (existing === undefined) buckets.push({ gram, count: 1 })
    else existing.count++
  }
  return buckets
}

export function referenceShared(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let shared = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) shared += Math.min(bucket.count, match.count)
  }
  return shared
}

export function referenceDot(
  a: readonly unknown[],
  b: readonly unknown[],
  gramSize: number,
): number {
  const right = referenceGrams(b, gramSize)
  let product = 0
  for (const bucket of referenceGrams(a, gramSize)) {
    const match = right.find((other) =>
      other.gram.every((element, index) => element === bucket.gram[index]),
    )
    if (match !== undefined) product += bucket.count * match.count
  }
  return product
}
