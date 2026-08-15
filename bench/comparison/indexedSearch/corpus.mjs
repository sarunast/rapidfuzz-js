// @ts-check
/**
 * The corpus the size ladder searches, shared by the two scripts that measure
 * it — `throughput.mjs` for time and `memory.mjs` for space. One
 * definition, because a memory figure that describes different strings from the
 * timing figure beside it is worse than no figure.
 */

/** xorshift32, so every run sees byte-identical input. */
function rng(seed) {
  let state = seed >>> 0
  return () => {
    state ^= state << 13
    state >>>= 0
    state ^= state >>> 17
    state ^= state << 5
    state >>>= 0
    return state / 0x1_0000_0000
  }
}

const LOWER = [...'abcdefghijklmnopqrstuvwxyz']

function word(next, length) {
  const characters = new Array(length)
  for (let index = 0; index < length; index++) {
    characters[index] = LOWER[Math.floor(next() * LOWER.length)]
  }
  return characters.join('')
}

/**
 * Sentences drawn from a Zipf-weighted vocabulary — a few words carry most of
 * the text, as in real prose. A uniform corpus would give every n-gram the same
 * posting length and flatter the index; it is skew that decides this.
 *
 * Because the generator is seeded and drawn in one pass, a smaller corpus is a
 * prefix of a larger one: the size ladder varies N and nothing else.
 *
 * @param {number} count
 * @returns {{ choices: string[], queries: [string, string][] }}
 */
export function buildCorpus(count) {
  const next = rng(0x0d15_ea5e)
  const vocabulary = []
  for (let index = 0; index < 400; index++) {
    vocabulary.push(word(next, 3 + Math.floor(next() * 6)))
  }
  const weights = vocabulary.map((_value, rank) => 1 / (rank + 1))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  const pick = () => {
    let target = next() * total
    for (let rank = 0; rank < weights.length; rank++) {
      target -= weights[rank]
      if (target <= 0) return vocabulary[rank]
    }
    return vocabulary[weights.length - 1]
  }
  const choices = []
  for (let index = 0; index < count; index++) {
    const parts = []
    for (let part = 0; part < 4; part++) parts.push(pick())
    // `join`, never `+=`: a rope built by concatenation flattens later and
    // charges the flattening to whatever ran next, which a heap delta reads as
    // the structure under measurement.
    choices.push(parts.join(' '))
  }
  const phrase = choices[Math.floor(count / 2)]
  const typo = (value) => {
    const characters = [...value]
    const at = Math.floor(characters.length / 2)
    characters[at] = characters[at] === 'a' ? 'b' : 'a'
    return characters.join('')
  }
  return {
    choices,
    queries: [
      ['whole phrase', phrase],
      ['one typo', typo(phrase)],
      ['half a phrase', phrase.split(' ').slice(0, 2).join(' ')],
      ['common word', vocabulary[0]],
      ['unrelated', 'qxzjvwkqxzjv'],
    ],
  }
}
