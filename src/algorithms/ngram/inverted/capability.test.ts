import { describe, expect, it } from 'vitest'

import { createScorer, scorerCompilation } from '#core/scoring/scorer.js'

import { exhaustive, METRICS, pairs } from '../../../../testing/invertedIndex.js'
import {
  distance as cosineDistance,
  similarity as cosineSimilarity,
} from '../../cosine/index.js'
import {
  distance as diceDistance,
  similarity as diceSimilarity,
} from '../../dice/index.js'

describe('the capability a metric declares', () => {
  it('is offered by both similarity metrics and answers like the Matcher', () => {
    const choices = ['abcd', 'abce', 'zzzz']
    for (const metric of METRICS) {
      const scorer = createScorer(metric === 'dice' ? diceSimilarity : cosineSimilarity, {
        gramSize: 3,
      })
      const indexChoices = scorerCompilation(scorer).indexChoices
      expect(indexChoices).toBeTypeOf('function')
      if (indexChoices === undefined) throw new Error('no index capability')
      const builder = indexChoices()
      for (const choice of choices) builder.add(choice)
      expect(pairs(builder.seal().select('abcd', 0.5, 3))).toEqual(
        exhaustive(metric, 3, choices, 'abcd', 0.5, 3),
      )
    }
  })

  it('is offered at the gram size the scorer was configured with', () => {
    const choices = ['abcd', 'abce']
    for (const gramSize of [2, 3, 4]) {
      const indexChoices = scorerCompilation(
        createScorer(diceSimilarity, { gramSize }),
      ).indexChoices
      if (indexChoices === undefined) throw new Error('no index capability')
      const builder = indexChoices()
      for (const choice of choices) builder.add(choice)
      expect(pairs(builder.seal().select('abcd', null, 2))).toEqual(
        exhaustive('dice', gramSize, choices, 'abcd', null, 2),
      )
    }
  })

  it('is absent on the distance direction', () => {
    expect(scorerCompilation(createScorer(diceDistance)).indexChoices).toBeUndefined()
    expect(scorerCompilation(createScorer(cosineDistance)).indexChoices).toBeUndefined()
  })
})
