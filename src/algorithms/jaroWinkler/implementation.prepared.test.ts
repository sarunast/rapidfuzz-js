import { describe, expect, it } from 'vitest'

import { PREPARE_SCORER } from '#core/scoring/builtIn/preparation.js'

import { jaroWinklerSimilarity } from './implementation.js'

const codes = (text: string): Uint16Array =>
  Uint16Array.from(text, (element) => element.charCodeAt(0))

// The prepared kernel converts a mixed query once per prepared query, so both
// orientations have to score exactly like the one-shot metric.
describe('prepared scoring across mixed representations', () => {
  const query = `${'a'.repeat(40)}${'b'.repeat(10)}`
  const choices = ['z'.repeat(50), `${'a'.repeat(40)}${'c'.repeat(10)}`, query]

  it('scores array choices from a string query like the one-shot metric', () => {
    const preparation = jaroWinklerSimilarity[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(query)
    for (const choice of choices) {
      expect(kernel(preparation.prepareChoice(codes(choice)), null)).toBe(
        jaroWinklerSimilarity(query, choice),
      )
    }
  })

  it('scores string choices from an array query like the one-shot metric', () => {
    const preparation = jaroWinklerSimilarity[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(codes(query))
    for (const choice of choices) {
      expect(kernel(preparation.prepareChoice(choice), null)).toBe(
        jaroWinklerSimilarity(codes(query), choice),
      )
    }
  })
})
