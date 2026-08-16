import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'
import { createIndexedMatcher, createMatcher } from '#search/index.js'

import {
  CORPORA,
  exhaustive,
  exhaustiveScan,
  indexOf,
  LIMITS,
  pairs,
  QUERIES,
  THRESHOLDS,
} from '../../../../testing/invertedIndex.js'
import { similarity as diceSimilarity } from '../../dice/index.js'
import { similarity as tverskySimilarity } from '../../tversky/index.js'
import { createTverskyIndexBuilder } from './tversky.js'

describe('the Tversky index at the Dice weights', () => {
  it('answers exactly what the Dice index and the exhaustive matcher answer', () => {
    // The wiring routes `.5/.5` to the Dice index, so the Tversky index never
    // serves these weights in production — this holds the implementation
    // correct at the configuration anyway, and `toEqual` makes it the
    // strongest differential in the file: bit-identical scores, ordering and
    // tie behaviour across dense, sparse, repeated and gramless shapes.
    for (const gramSize of [1, 2, 3]) {
      for (const choices of CORPORA) {
        for (const query of QUERIES) {
          const viaTversky = createTverskyIndexBuilder(gramSize, 0.5, 0.5)
          for (const choice of choices) viaTversky.add(choice)
          const tversky = viaTversky.seal()
          const dice = indexOf({ metric: 'dice' }, gramSize, choices)
          for (const threshold of THRESHOLDS) {
            for (const limit of LIMITS) {
              const found = pairs(tversky.select(query, threshold, limit))
              expect(found).toEqual(pairs(dice.select(query, threshold, limit)))
              expect(found).toEqual(
                exhaustive(
                  { metric: 'dice' },
                  gramSize,
                  choices,
                  query,
                  threshold,
                  limit,
                ),
              )
            }
            const scanned = pairs(tversky.scan(query, threshold))
            expect(scanned).toEqual(pairs(dice.scan(query, threshold)))
            expect(scanned).toEqual(
              exhaustiveScan({ metric: 'dice' }, gramSize, choices, query, threshold),
            )
          }
        }
      }
    }
  })
})

describe('the weights change the ranking, not just the scores', () => {
  // One query against a candidate that misses a piece and a candidate that
  // contains everything plus extras. Which one wins is exactly the α/β
  // orientation — a reversed `queryGrams/choiceGrams` in the index formula
  // would pass every symmetric test and fail here.
  const query = 'new york mets'
  const missing = 'new york'
  const containing = 'the wonderful new york mets'
  const choices = [missing, containing]

  it('lets the containing candidate win when only query coverage counts', () => {
    const index = indexOf({ metric: 'tversky', alpha: 1, beta: 0 }, 2, choices)
    const found = pairs(index.select(query, null, 2))
    expect(found).toEqual(
      exhaustive({ metric: 'tversky', alpha: 1, beta: 0 }, 2, choices, query, null, 2),
    )
    expect(found[0]).toEqual({ id: 1, score: 1 })
  })

  it('punishes the extras when the choice side carries the weight', () => {
    const index = indexOf({ metric: 'tversky', alpha: 0.1, beta: 1 }, 2, choices)
    const found = pairs(index.select(query, null, 2))
    expect(found).toEqual(
      exhaustive({ metric: 'tversky', alpha: 0.1, beta: 1 }, 2, choices, query, null, 2),
    )
    expect(found[0].id).toBe(0)
  })
})

describe('the public indexed matcher', () => {
  it('matches the exhaustive matcher at huge finite weights', () => {
    // `alpha * grams` alone would overflow to Infinity; the index precomputes
    // its scaled coefficients at construction, and this is the regression that
    // notices a naive copy of the formula.
    const choices = ['abcdef', 'abcxyz', 'uvwxyz']
    for (const weights of [
      { alpha: Number.MAX_VALUE, beta: Number.MAX_VALUE },
      { alpha: Number.MAX_VALUE, beta: 0 },
    ]) {
      const scorer = createScorer(tverskySimilarity, weights)
      const indexed = createIndexedMatcher(choices, { scorer })
      const exhaustiveMatcher = createMatcher(choices, { scorer })
      for (const query of ['abcdef', 'bana', 'abcxyz']) {
        expect(indexed.search(query, { limit: null })).toEqual(
          exhaustiveMatcher.search(query, { limit: null }),
        )
      }
    }
  })

  it('serves the default weights through the Dice index, indistinguishably', () => {
    // `.5/.5` routes to the Dice index builder; the answers have to be the
    // ones a Dice scorer's own index gives, scores bit-identical.
    const choices = ['banana', 'bananas', 'band', 'ananab', 'qq']
    const viaTversky = createIndexedMatcher(choices, {
      scorer: createScorer(tverskySimilarity, { gramSize: 3, alpha: 0.5, beta: 0.5 }),
    })
    const viaDice = createIndexedMatcher(choices, {
      scorer: createScorer(diceSimilarity, { gramSize: 3 }),
    })
    for (const query of ['banana', 'ban', 'zzz']) {
      const found = viaTversky.search(query, { limit: null })
      expect(found.map((match) => ({ key: match.key, score: match.score }))).toEqual(
        viaDice
          .search(query, { limit: null })
          .map((match) => ({ key: match.key, score: match.score })),
      )
    }
  })

  it('serves an asymmetric configuration end to end', () => {
    const scorer = createScorer(tverskySimilarity, { gramSize: 3, alpha: 1, beta: 0.1 })
    const choices = ['banana', 'bananas', 'band', 'ananab', 'qq']
    const indexed = createIndexedMatcher(choices, { scorer })
    const exhaustiveMatcher = createMatcher(choices, { scorer })
    for (const query of ['banana', 'ban', 'zzz']) {
      expect(indexed.best(query)).toEqual(exhaustiveMatcher.best(query))
      expect(indexed.search(query, { limit: 3, threshold: 0.1 })).toEqual(
        exhaustiveMatcher.search(query, { limit: 3, threshold: 0.1 }),
      )
    }
  })
})
