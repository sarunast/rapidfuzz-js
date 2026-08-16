import { describe, expect, it } from 'vitest'

import { PREPARE_SCORER } from '#core/scoring/builtIn/preparation.js'

import { levenshteinDistance } from './metric.js'

const codes = (text: string): Uint16Array =>
  Uint16Array.from(text, (element) => element.charCodeAt(0))

// The prepared kernel probes mixed representations without aligning them, so
// both orientations have to dispatch and score exactly like the one-shot
// metric: an affixed pair long enough to be worth trimming takes the trimmed
// dynamic program, an unrelated one stays on the prepared kernel.
describe('prepared scoring across mixed representations', () => {
  const query = `${'a'.repeat(100)}${'b'.repeat(28)}`
  const choices = {
    unrelated: 'z'.repeat(128),
    'large prefix': `${'a'.repeat(100)}${'c'.repeat(28)}`,
    'large suffix': `${'c'.repeat(28)}${'a'.repeat(72)}${'b'.repeat(28)}`,
  }

  it('scores array choices from a string query like the one-shot metric', () => {
    const preparation = levenshteinDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(query)
    for (const choice of Object.values(choices)) {
      expect(kernel(preparation.prepareChoice(codes(choice)), null)).toBe(
        levenshteinDistance(query, choice),
      )
      expect(kernel(preparation.prepareChoice(codes(choice)), 100)).toBe(
        levenshteinDistance(query, choice, { scoreCutoff: 100 }),
      )
    }
  })

  it('scores string choices from an array query like the one-shot metric', () => {
    const preparation = levenshteinDistance[PREPARE_SCORER]({})
    const kernel = preparation.prepareQuery(codes(query))
    for (const choice of Object.values(choices)) {
      expect(kernel(preparation.prepareChoice(choice), null)).toBe(
        levenshteinDistance(codes(query), choice),
      )
      expect(kernel(preparation.prepareChoice(choice), 100)).toBe(
        levenshteinDistance(codes(query), choice, { scoreCutoff: 100 }),
      )
    }
  })
})
