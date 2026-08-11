import type { Sequence } from './types.js'

const NON_ALNUM = /[^\p{L}\p{N}]/gu

/**
 * JavaScript lowercases with Unicode's full mapping, RapidFuzz with a simple
 * per-code-point table. `İ` gives `i` + `U+0307` where upstream gives `i`, and
 * a word-final `Σ` gives `ς` where upstream always gives `σ`.
 *
 * Replaced before lowercasing, never repaired after: a dot or a final sigma the
 * caller wrote is upstream's own answer, and by then the two are the same
 * character. Verified against rapidfuzz 3.14.5 and Node 26.5.1 on 2026-08-11 —
 * these two, and no others, over every code point. A finding about two
 * versions; re-run the sweep rather than trusting the class.
 */
const FULL_CASE_DIVERGENT = /[İΣ]/
const DOTTED_CAPITAL_I = 'İ'
const CAPITAL_SIGMA = 'Σ'

/**
 * Lowercase, replace every non-alphanumeric character with a space, and trim.
 *
 * Follows RapidFuzz's `utils.default_process`: an underscore separates, runs
 * are not collapsed (`'a---b'` → `'a   b'`), and no `NFC`/`NFKC` is applied.
 *
 * Takes a {@link Sequence} though it accepts only strings, because it is the
 * package's `Normalizer` and a choice may be an array. Narrowing to `string`
 * makes it stop being assignable to the type it exists to satisfy.
 */
export function normalizeText(value: Sequence): string {
  if (typeof value !== 'string') throw new TypeError('normalizeText expects a string')
  const separated = value.replace(NON_ALNUM, ' ').trim()
  // One scan, so ordinary input pays no replacement pass at all.
  if (!FULL_CASE_DIVERGENT.test(separated)) return separated.toLowerCase()
  return separated
    .replaceAll(DOTTED_CAPITAL_I, 'i')
    .replaceAll(CAPITAL_SIGMA, 'σ')
    .toLowerCase()
}
