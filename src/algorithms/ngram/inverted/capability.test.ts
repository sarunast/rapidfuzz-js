import { describe, expect, it } from 'vitest'

import { createScorer, scorerCompilation } from '#core/scoring/scorer.js'

import {
  exhaustive,
  pairs,
  REPRESENTATION_SPECS,
  type MetricSpec,
} from '../../../../testing/invertedIndex.js'
import {
  distance as cosineDistance,
  similarity as cosineSimilarity,
} from '../../cosine/index.js'
import {
  distance as diceDistance,
  similarity as diceSimilarity,
} from '../../dice/index.js'
import {
  distance as tverskyDistance,
  similarity as tverskySimilarity,
} from '../../tversky/index.js'

function similarityScorerOf(spec: MetricSpec, gramSize: number) {
  switch (spec.metric) {
    case 'dice':
      return createScorer(diceSimilarity, { gramSize })
    case 'cosine':
      return createScorer(cosineSimilarity, { gramSize })
    case 'tversky':
      return createScorer(tverskySimilarity, {
        gramSize,
        alpha: spec.alpha,
        beta: spec.beta,
      })
  }
}

describe('the capability a metric declares', () => {
  it('is offered by every similarity metric and answers like the Matcher', () => {
    const choices = ['abcd', 'abce', 'zzzz']
    for (const spec of REPRESENTATION_SPECS) {
      const indexChoices = scorerCompilation(similarityScorerOf(spec, 3)).indexChoices
      expect(indexChoices).toBeTypeOf('function')
      if (indexChoices === undefined) throw new Error('no index capability')
      const builder = indexChoices()
      for (const choice of choices) builder.add(choice)
      expect(pairs(builder.seal().select('abcd', 0.5, 3))).toEqual(
        exhaustive(spec, 3, choices, 'abcd', 0.5, 3),
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
        exhaustive({ metric: 'dice' }, gramSize, choices, 'abcd', null, 2),
      )
    }
  })

  it('captures the weights the scorer was configured with', () => {
    // α/β are closed over by `indexChoices`; scoring `abcd` against a corpus
    // that contains it separates containment from the defaults, which would
    // dock the longer choice for its extra grams.
    const choices = ['abcdefgh', 'zzzz']
    const spec: MetricSpec = { metric: 'tversky', alpha: 1, beta: 0 }
    const indexChoices = scorerCompilation(
      createScorer(tverskySimilarity, { gramSize: 3, alpha: 1, beta: 0 }),
    ).indexChoices
    if (indexChoices === undefined) throw new Error('no index capability')
    const builder = indexChoices()
    for (const choice of choices) builder.add(choice)
    const found = pairs(builder.seal().select('abcd', null, 2))
    expect(found).toEqual(exhaustive(spec, 3, choices, 'abcd', null, 2))
    expect(found[0]).toEqual({ id: 0, score: 1 })
  })

  it('is absent on the distance direction', () => {
    expect(scorerCompilation(createScorer(diceDistance)).indexChoices).toBeUndefined()
    expect(scorerCompilation(createScorer(cosineDistance)).indexChoices).toBeUndefined()
    expect(
      scorerCompilation(createScorer(tverskyDistance, { alpha: 1, beta: 0.1 }))
        .indexChoices,
    ).toBeUndefined()
  })
})
