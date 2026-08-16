import { describe, expect, it } from 'vitest'

import { PREPARE_SCORER } from '#core/scoring/builtIn/preparation.js'

import { osaDistance } from './implementation.js'

const codes = (text: string): Uint16Array =>
  Uint16Array.from(text, (element) => element.charCodeAt(0))

// The trimming fallback runs only when the query is the longer side, so these
// choices are shorter than the query; the mixed query converts once per
// prepared query and both orientations score like the one-shot metric.
describe('prepared scoring across mixed representations', () => {
  const query = `${'a'.repeat(50)}${'b'.repeat(14)}`
  const choices = ['z'.repeat(50), `${'a'.repeat(40)}${'c'.repeat(10)}`]

  it('scores array choices from a string query like the one-shot metric', () => {
    const preparation = osaDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(query)
    for (const choice of choices) {
      expect(kernel(preparation.prepareChoice(codes(choice)), null)).toBe(
        osaDistance(query, choice),
      )
    }
  })

  it('scores string choices from an array query like the one-shot metric', () => {
    const preparation = osaDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(codes(query))
    for (const choice of choices) {
      expect(kernel(preparation.prepareChoice(choice), null)).toBe(
        osaDistance(codes(query), choice),
      )
    }
  })
})
