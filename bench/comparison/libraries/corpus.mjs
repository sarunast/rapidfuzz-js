// @ts-check
/**
 * The corpus both legs of the comparison run against.
 *
 * Deliberately separate from `bench/harness/corpus.ts`, which is fingerprinted into
 * every baseline entry — editing that file costs a full re-record, and this one
 * has nothing to do with the baseline suite. It is also `.mjs` rather than
 * `.ts` because the Python leg reads the same data through a JSON file, and a
 * generator that needs a build step first cannot be the thing that writes it.
 *
 * Generated from a seeded xorshift rather than `Math.random`, so a run is
 * comparable against another run, and so the JavaScript and Python legs see
 * byte-identical inputs.
 */

const LOWERCASE = 'abcdefghijklmnopqrstuvwxyz'

/**
 * xorshift32. The same generator `bench/harness/corpus.ts` uses, restated here rather
 * than imported, because that file is fingerprinted and this one must not
 * depend on it.
 *
 * @param {number} seed
 * @returns {() => number} a generator of values in `[0, 1)`
 */
function seeded(seed) {
  let state = seed | 0 || 1
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / 4294967296
  }
}

/**
 * @param {() => number} random
 * @param {number} length
 * @returns {string}
 */
function word(random, length) {
  let out = ''
  for (let i = 0; i < length; i++) {
    out += LOWERCASE[Math.floor(random() * LOWERCASE.length)]
  }
  return out
}

/**
 * A pair that differs in a few positions, which is the shape fuzzy matching is
 * actually asked about — two strings with no elements in common exercise only
 * the early exits.
 *
 * @param {() => number} random
 * @param {number} length
 * @returns {[string, string]}
 */
function editedPair(random, length) {
  const first = word(random, length)
  const edits = Math.max(1, Math.floor(length / 16))
  const characters = [...first]
  for (let i = 0; i < edits; i++) {
    const at = Math.floor(random() * characters.length)
    characters[at] = LOWERCASE[Math.floor(random() * LOWERCASE.length)]
  }
  return [first, characters.join('')]
}

/**
 * @typedef {object} Corpus
 * @property {Record<string, [string, string][]>} pairs      keyed by length
 * @property {[string, string][]}                 sentences  word-like phrases
 * @property {string[]}                           choices    the haystack
 * @property {string[]}                           queries    needles for it
 * @property {string[]}                           matrixRows queries for a matrix
 * @property {string[]}                           matrixCols choices for it
 * @property {string[]}                           titles     multiword search choices
 * @property {string[]}                           titleQueries multiword needles
 */

/** The lengths the distance tasks run at. */
export const PAIR_LENGTHS = [8, 32, 128, 1024]

/**
 * Build the corpus. Same seed, same data, every run and every language.
 *
 * @returns {Corpus}
 */
export function buildCorpus() {
  const random = seeded(0x9e3779b9)

  /** @type {Record<string, [string, string][]>} */
  const pairs = {}
  for (const length of PAIR_LENGTHS) {
    // Fewer long pairs: a 1024-character comparison is ~16,000 times the work
    // of an 8-character one, and the point is a comparable wall clock per task.
    const count = length >= 1024 ? 25 : 200
    /** @type {[string, string][]} */
    const built = []
    for (let i = 0; i < count; i++) built.push(editedPair(random, length))
    pairs[String(length)] = built
  }

  /** @type {[string, string][]} */
  const sentences = []
  for (let i = 0; i < 200; i++) {
    const words = 4 + Math.floor(random() * 5)
    /** @type {string[]} */
    const left = []
    for (let w = 0; w < words; w++) left.push(word(random, 3 + Math.floor(random() * 6)))
    const right = [...left]
    right[Math.floor(random() * right.length)] = word(random, 5)
    sentences.push([left.join(' '), right.join(' ')])
  }

  /** @type {string[]} */
  const choices = []
  for (let i = 0; i < 2000; i++) choices.push(word(random, 8 + Math.floor(random() * 8)))

  /** @type {string[]} */
  const queries = []
  for (let i = 0; i < 20; i++) queries.push(word(random, 8 + Math.floor(random() * 8)))

  const matrixRows = choices.slice(0, 50)
  const matrixCols = choices.slice(50, 250)

  const titles = []
  for (let i = 0; i < 2000; i++) {
    const parts = []
    for (let w = 0; w < 5; w++) parts.push(word(random, 3 + Math.floor(random() * 6)))
    titles.push(parts.join(' '))
  }

  const titleQueries = []
  for (let i = 0; i < 20; i++) {
    const parts = []
    for (let w = 0; w < 5; w++) parts.push(word(random, 3 + Math.floor(random() * 6)))
    titleQueries.push(parts.join(' '))
  }

  return {
    pairs,
    sentences,
    choices,
    queries,
    matrixRows,
    matrixCols,
    titles,
    titleQueries,
  }
}
