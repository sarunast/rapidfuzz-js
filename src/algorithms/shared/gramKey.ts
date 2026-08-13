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
