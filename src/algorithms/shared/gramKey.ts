/**
 * The rungs a packed gram key can sit on, narrowest first: a byte for Latin-1,
 * a BMP word, and the full code-point range.
 */
const RADIX_LADDER: readonly number[] = [0x100, 0x1_0000, 0x11_0000]

/**
 * The radices that hold a gram of this depth inside one safe integer, narrowest
 * first. Latin-1 text needs 8 bits an element, so `'abc'` packs into 24 —
 * `0x616263` — where a BMP radix spends 48 on the same three letters, and small
 * integer keys are the ones a `Map` handles best.
 *
 * Depth decides how far the ladder reaches: a byte radix holds six elements, a
 * BMP radix three, the full code-point radix two. A trigram over astral text
 * therefore has no packed rung at all and falls back to joined strings.
 */
export function feasibleRadices(gramSize: number): readonly number[] {
  return RADIX_LADDER.filter(
    (radix) => Math.pow(radix, gramSize) <= Number.MAX_SAFE_INTEGER,
  )
}

/**
 * The one radix a prepared profile packs at, or `null` where the ladder has no
 * rung deep enough — `0x110000` up to bigrams, `0x10000` at trigrams, `0x100`
 * up to six elements, nothing beyond.
 *
 * The **widest** feasible rung, where an index takes the narrowest: an index
 * starts small and re-keys itself wider when a build demands it, while two
 * profiles meet with no shared context to re-key against. One canonical radix
 * per depth is what lets them compare keys at all — the same gram spelled at
 * two radices is two different numbers.
 *
 * Spelled out rather than read off `feasibleRadices`, which allocates an array
 * and raises three powers to reach a constant — this runs once per profile
 * built and once per direct comparison. It agrees with the ladder by test
 * rather than by construction, so a rung added there without a case here is a
 * failure.
 */
export function canonicalRadix(gramSize: number): number | null {
  if (gramSize <= 2) return 0x11_0000
  if (gramSize === 3) return 0x1_0000
  if (gramSize <= 6) return 0x100
  return null
}

/**
 * A gram as one integer, most significant element first. Positional and so
 * reversible by {@link unpackGram}, which is what lets a packed profile answer
 * a trie one.
 */
export function packGram(
  digits: ArrayLike<number>,
  start: number,
  gramSize: number,
  radix: number,
): number {
  let key = 0
  for (let offset = 0; offset < gramSize; offset++)
    key = key * radix + digits[start + offset]
  return key
}

/** {@link packGram} reversed, writing `gramSize` digits into `digits`. */
export function unpackGram(
  key: number,
  gramSize: number,
  radix: number,
  digits: number[],
): void {
  let rest = key
  for (let offset = gramSize - 1; offset >= 0; offset--) {
    digits[offset] = rest % radix
    rest = Math.floor(rest / radix)
  }
}
