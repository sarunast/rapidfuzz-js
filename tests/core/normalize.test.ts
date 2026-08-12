// Ported from RapidFuzz tests/test_utils.py
import { expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'

it('preprocesses mixed unicode strings', () => {
  const mixed: ReadonlyArray<readonly [string, string]> = [
    [
      'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
      'lorem ipsum is simply dummy text of the printing and typesetting industry',
    ],
    ["C'est la vie", 'c est la vie'],
    ['Ça va?', 'ça va'],
    ['Cães danados', 'cães danados'],
    ['¬Camarões assados', 'camarões assados'],
    ['a¬ሴ€耀', 'a ሴ 耀'],
    ['Á', 'á'],
  ]

  for (const [input, expected] of mixed) {
    expect(defaultProcess(input)).toBe(expected)
  }
})

// rapidfuzz 3.14.5, 2026-08-11: default_process('İstanbul') -> 'istanbul', a
// bare 'i'. JavaScript answers 'i̇stanbul', the full case mapping adding a
// combining dot. The only per-code-point disagreement out of 147,596.
it('lowercases the dotted capital I as upstream does', () => {
  expect(defaultProcess('İstanbul')).toBe('istanbul')
  expect([...defaultProcess('İ')]).toEqual(['i'])
  expect(defaultProcess('DİYARBAKIR İZMİR')).toBe('diyarbakir izmir')
  // A combining dot that was in the input is not alphanumeric, so it is a
  // separator before any of this — the fix must not be a blanket strip.
  expect(defaultProcess('café')).toBe('cafe')
  expect(defaultProcess('i̇')).toBe('i')
})

// rapidfuzz 3.14.5, 2026-08-11: default_process('ΟΔΟΣ ΟΔΟΣ') -> 'οδοσ οδοσ'.
// JavaScript answers 'οδος οδος', its full case mapping making a word-final
// sigma. A per-code-point sweep cannot find this one — a lone 'Σ' agrees.
it('lowercases a word-final capital sigma as upstream does', () => {
  expect(defaultProcess('ΟΔΟΣ ΟΔΟΣ')).toBe('οδοσ οδοσ')
  expect(defaultProcess('ΣΊΣΥΦΟΣ')).toBe('σίσυφοσ')
  expect(defaultProcess('ΑΣ')).toBe('ασ')
  expect(defaultProcess('ΑΣΑ')).toBe('ασα')
  expect(defaultProcess('ΣΣΣ')).toBe('σσσ')
  // A final sigma the caller wrote is upstream's own answer and survives — the
  // reason the repair runs before lowercasing rather than as a fold after it.
  expect(defaultProcess('οδος')).toBe('οδος')
  expect(defaultProcess('ΟΔΟΣ')).not.toBe(defaultProcess('οδος'))
})

// The rest of `default_process`, asserted against rapidfuzz 3.14.5 rather than
// assumed: one replacement per character, by Unicode letter/number class.
it('follows default_process on separators, runs, and case classes', () => {
  const cases: ReadonlyArray<readonly [string, string]> = [
    ['a---b', 'a   b'],
    ['  padded  ', 'padded'],
    ['snake_case_name', 'snake case name'],
    ['tab\ttab', 'tab tab'],
    ['', ''],
    ['_', ''],
    ['ABC123def', 'abc123def'],
    // Compatibility characters are letters and numbers: no NFKC here.
    ['ﬁ', 'ﬁ'],
    ['³²¹', '³²¹'],
    ['𝟙𝟚', '𝟙𝟚'],
    // Lowercasing the two agree on, titlecase letter included.
    ['ǅungla', 'ǆungla'],
    ['Ⅻ', 'ⅻ'],
    ['ß', 'ß'],
  ]
  for (const [input, expected] of cases) {
    expect(defaultProcess(input)).toBe(expected)
  }
})

it('returns a non-text sequence as it came', () => {
  // An array typechecks here on purpose: `normalizeText` is a `Normalizer`, and
  // a `Normalizer` is handed whatever a choice turns out to be. Nothing about
  // an element is text to lowercase, so the sequence itself comes back — `toBe`,
  // because a copy would be a different answer to the same question.
  const elements = ['a', 'b']
  expect(defaultProcess(elements)).toBe(elements)
  const arrayLike = { length: 1, 0: 'A' }
  expect(defaultProcess(arrayLike)).toBe(arrayLike)
})

it('rejects what is not a sequence at all', () => {
  // Through `Reflect.apply`, because the types refuse these and the project
  // bans the casts that would silence them.
  for (const invalid of [1, {}, null, undefined, true, () => 'a']) {
    expect(() => Reflect.apply(defaultProcess, undefined, [invalid])).toThrow(
      'expected a string or an array-like sequence',
    )
  }
})
