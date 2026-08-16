import { describe, expect, it } from 'vitest'

import { PREPARE_SCORER } from '#core/scoring/builtIn/preparation.js'

import { indelDistance } from './implementation.js'

const codes = (text: string): Uint16Array =>
  Uint16Array.from(text, (element) => element.charCodeAt(0))

// The prepared kernel probes mixed representations without aligning them, so
// both orientations have to dispatch and score exactly like the one-shot
// metric: an affixed pair takes the trimming fallback, an unrelated one stays
// on the prepared kernel.
describe('prepared scoring across mixed representations', () => {
  const query = `${'a'.repeat(90)}${'b'.repeat(10)}`
  const choices = {
    unrelated: 'z'.repeat(50),
    'large prefix': `${'a'.repeat(40)}${'c'.repeat(10)}`,
    'large suffix': `${'c'.repeat(10)}${'a'.repeat(30)}${'b'.repeat(10)}`,
  }

  it('scores array choices from a string query like the one-shot metric', () => {
    const preparation = indelDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(query)
    for (const choice of Object.values(choices)) {
      expect(kernel(preparation.prepareChoice(codes(choice)), null)).toBe(
        indelDistance(query, choice),
      )
      expect(kernel(preparation.prepareChoice(codes(choice)), 60)).toBe(
        indelDistance(query, choice, { scoreCutoff: 60 }),
      )
    }
  })

  it('scores string choices from an array query like the one-shot metric', () => {
    const preparation = indelDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(codes(query))
    for (const choice of Object.values(choices)) {
      expect(kernel(preparation.prepareChoice(choice), null)).toBe(
        indelDistance(codes(query), choice),
      )
      expect(kernel(preparation.prepareChoice(choice), 60)).toBe(
        indelDistance(codes(query), choice, { scoreCutoff: 60 }),
      )
    }
  })
})
