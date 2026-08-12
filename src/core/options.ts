/**
 * Refuses an own enumerable key the option bag does not define — what a caller
 * writes in an object literal, which is the misspelling worth catching. A
 * non-enumerable or inherited one passes, and `Object.keys` is why: a key
 * someone put on `Object.prototype` must not make every valid call throw.
 *
 * Only bags with optional behaviour are checked: a misspelled `threshold` or
 * `normalize` silently turns that behaviour off, where a bag whose keys are all
 * required fails on the missing one instead.
 *
 * Reads `unknown` because a JavaScript caller's `null` gets here, and
 * `Object.keys(null)` throws an error about our internals.
 */
export function assertOptionKeys(
  given: unknown,
  known: readonly string[],
  label: string,
): void {
  if (typeof given !== 'object' || given === null) {
    throw new TypeError(`${label} options must be an object`)
  }
  for (const key of Object.keys(given)) {
    if (!known.includes(key)) {
      throw new TypeError(`unknown ${label} option '${key}'`)
    }
  }
}
