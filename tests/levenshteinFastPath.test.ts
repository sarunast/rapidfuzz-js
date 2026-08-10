// Not a port. `levenshteinDistance` answers the call shape it is given most
// often — two BMP strings and no options — without building the options,
// weights and cutoff machinery the general case needs, which means there are
// two implementations of the same function and nothing in the ported suite
// compares them against each other.
//
// Passing `{}` is what makes that comparison possible: it is semantically the
// same call, and it is enough to send it down the general path. Every
// assertion below is that the two agree.
import fc from 'fast-check'
import { expect, test } from 'vitest'

import { levenshteinDistance } from '../src/distance/levenshtein.js'

const same = (a: string, b: string): void => {
  expect(levenshteinDistance(a, b)).toBe(levenshteinDistance(a, b, {}))
}

const fromAlphabet = (
  alphabet: readonly string[],
  maxLength: number,
): fc.Arbitrary<string> =>
  fc.array(fc.constantFrom(...alphabet), { maxLength }).map((chars) => chars.join(''))

test('the fast path agrees with the general path on arbitrary strings', () => {
  fc.assert(fc.property(fc.string(), fc.string(), same), { numRuns: 20_000 })
})

// Which representation a pair takes is decided by whether either side holds a
// surrogate pair, so the alphabet has to be able to build one, half of one, and
// none at all — a lone surrogate is not a pair and must stay on the fast path.
test('the fast path agrees on strings that straddle the BMP', () => {
  const tricky = fromAlphabet(['a', 'b', '\u{1f600}', '\ud800', '\udc00', 'é', 'ю'], 80)
  fc.assert(fc.property(tricky, tricky, same), { numRuns: 30_000 })
})

test('the fast path agrees at every kernel boundary', () => {
  // One word, two words, and either side of each — the lengths at which
  // `levenshteinUniform` changes which kernel answers.
  const lengths = [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000]
  const units = ['a', 'ab', 'é', '\u{1f600}', '\ud800']
  const cases: string[] = []
  for (const unit of units) for (const length of lengths) cases.push(unit.repeat(length))

  for (const a of cases) for (const b of cases) same(a, b)
})
