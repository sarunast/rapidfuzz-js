import type { Sequence } from './_common.js'

// Python's `\W` under `re.UNICODE` is "not a word character", where a word
// character is any letter, any number, or an underscore.
const NON_ALNUM = /[^\p{L}\p{N}_]/gu

/**
 * Preprocess a string by replacing every non-alphanumeric character with a
 * space, trimming, and lowercasing.
 *
 * Accepts {@link Sequence} so it can be passed directly as a scorer's
 * `processor`, but only strings are meaningful — anything else throws, the same
 * way RapidFuzz's Python implementation fails on a non-string.
 *
 * @example
 * defaultProcess('Lorem Ipsum, dolor!') // => 'lorem ipsum  dolor'
 *
 * @throws if `sentence` is not a string.
 */
export function defaultProcess(sentence: Sequence): string {
  if (typeof sentence !== 'string') {
    throw new TypeError('defaultProcess expects a string')
  }

  return sentence.replace(NON_ALNUM, ' ').trim().toLowerCase()
}
