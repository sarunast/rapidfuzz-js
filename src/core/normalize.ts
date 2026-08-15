import { validateSequence } from './sequence.js'
import type { Sequence } from './types.js'

const NON_ALNUM = /[^\p{L}\p{N}]/gu

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
  if (typeof value !== 'string') return validateSequence(value)
  const separated = value.replace(NON_ALNUM, ' ').trim()
  if (!FULL_CASE_DIVERGENT.test(separated)) return separated.toLowerCase()
  return separated
    .replaceAll(DOTTED_CAPITAL_I, 'i')
    .replaceAll(CAPITAL_SIGMA, 'σ')
    .toLowerCase()
}
