import { validateSequence } from './sequence.js'
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
 * ```ts
 * normalizeText('  Wireless-Mechanical KEYBOARD!! ') // 'wireless mechanical keyboard'
 * ```
 *
 * Follows RapidFuzz's `utils.default_process`: an underscore separates, runs
 * are not collapsed (`'a---b'` → `'a   b'`), and no `NFC`/`NFKC` is applied.
 * Like upstream it is opt-in — nothing applies it for you, because scoring what
 * you were given is the only honest default and cleaning is a decision about
 * your data.
 *
 * Pass it as the `normalize` option so it reaches both the choices and the
 * query; applying it by hand to only one side is the classic mistake.
 *
 * Normalizes text; any other sequence is returned as it came. That is what
 * makes it the package's `Normalizer` for a collection of array-like choices,
 * where nothing about an element is text to lowercase.
 *
 * @returns The cleaned string, or the sequence unchanged if it was not text.
 * @throws `TypeError` if the value is not a sequence at all — a number, a
 * boolean, or an object with no array-like `length`.
 */
export function normalizeText(value: string): string
export function normalizeText<TSequence extends ArrayLike<unknown>>(
  value: TSequence,
): TSequence
export function normalizeText(value: Sequence): Sequence {
  // Returned through the guard rather than directly: this is a public entry
  // point, and what it hands back has to be a sequence whether or not it had
  // text to normalize. `validateSequence` answers the value it was given.
  if (typeof value !== 'string') return validateSequence(value)
  const separated = value.replace(NON_ALNUM, ' ').trim()
  // One scan, so ordinary input pays no replacement pass at all.
  if (!FULL_CASE_DIVERGENT.test(separated)) return separated.toLowerCase()
  return separated
    .replaceAll(DOTTED_CAPITAL_I, 'i')
    .replaceAll(CAPITAL_SIGMA, 'σ')
    .toLowerCase()
}
