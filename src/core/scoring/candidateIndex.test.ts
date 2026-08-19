import { describe, expect, it } from 'vitest'

import {
  distance as diceDistance,
  similarity as diceSimilarity,
} from '#algorithms/dice/index.js'

import { createScorer, scorerCompilation } from './scorer.js'

describe('candidate index capability', () => {
  it('adapts exact similarity indexes, but never distance indexes', () => {
    expect(scorerCompilation(createScorer(diceSimilarity)).candidateChoices).toBeTypeOf(
      'function',
    )
    expect(scorerCompilation(createScorer(diceDistance)).candidateChoices).toBeUndefined()
  })

  it('adapts an exact index to the ids its own scan returns', () => {
    const compilation = scorerCompilation(createScorer(diceSimilarity))
    const make = compilation.candidateChoices
    const exact = compilation.indexChoices
    if (make === undefined || exact === undefined) {
      throw new Error('missing Dice candidate capability')
    }
    const corpus = ['night', 'nacht', 'naght', 'day', 'nigth']
    const builder = make()
    const reference = exact()
    for (const value of corpus) {
      builder.add(value)
      reference.add(value)
    }
    const index = builder.seal()
    const scan = reference.seal()

    for (const query of ['night', 'nacht', 'unrelated', '']) {
      for (const threshold of [0, 0.25, 0.5, 1]) {
        const found = index.candidates(query, threshold)
        const expected = scan.scan(query, threshold)
        expect(Array.from(found.ids.subarray(0, found.length))).toEqual(
          Array.from(expected.ids.subarray(0, expected.length)),
        )
      }
    }
    expect(() => builder.add('late')).toThrow(TypeError)
    expect(() => builder.seal()).toThrow(TypeError)
  })
})
