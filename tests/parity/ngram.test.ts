// Oracle: scripts/generate-ngram-fixtures.py, an independent `collections.Counter`
// reference. RapidFuzz ships neither of these, so there is nothing upstream to
// port; the formula is the specification, and the Python is it written out.
import { describe, expect, test } from 'vitest'

import * as cosine from '../../src/algorithms/cosine/index.js'
import * as dice from '../../src/algorithms/dice/index.js'
import { scoreMatrix, scorePairs } from '../../src/batch/index.js'
import type { Metric } from '../../src/core/metric.js'
import { createScorer } from '../../src/core/scorer.js'
import { bestMatch, createMatcher, search, searchIter } from '../../src/search/index.js'
import fixture from '../fixtures/ngram-oracle.json' with { type: 'json' }

interface NGramConfiguration {
  readonly gramSize?: number | undefined
}

interface NGramSuite {
  readonly distance: Metric<'distance', NGramConfiguration>
  readonly similarity: Metric<'similarity', NGramConfiguration>
  readonly normalizedDistance: Metric<'distance', NGramConfiguration>
  readonly normalizedSimilarity: Metric<'similarity', NGramConfiguration>
}

const FAMILIES: ReadonlyArray<readonly [string, NGramSuite, (typeof fixture)['dice']]> = [
  ['Sørensen-Dice', dice, fixture.dice],
  ['Cosine', cosine, fixture.cosine],
]

for (const [name, suite, cases] of FAMILIES) {
  describe(`${name} matches the n-gram oracle`, () => {
    test.each(cases)(
      'case %# at gramSize $gramSize',
      ({ left, right, gramSize, similarity }) => {
        const scorer = createScorer(suite.similarity, { gramSize })
        const distance = createScorer(suite.distance, { gramSize })

        expect(scorer.score(left, right)).toBeCloseTo(similarity, 12)
        expect(scorer.score(right, left)).toBeCloseTo(similarity, 12)
        expect(distance.score(left, right)).toBeCloseTo(1 - similarity, 12)

        // Every execution path has to agree with the direct one: the prepared
        // kernel reads gram counts and norms off the profile rather than
        // recomputing them, which is where the two could drift apart.
        expect(createMatcher([right], { scorer }).best(left)?.score).toBeCloseTo(
          similarity,
          12,
        )
        expect(bestMatch(left, [right], { scorer })?.score).toBeCloseTo(similarity, 12)
        expect(search(left, [right], { scorer, limit: null })[0]?.score).toBeCloseTo(
          similarity,
          12,
        )
        expect(Array.from(searchIter(left, [right], { scorer }))[0]?.score).toBeCloseTo(
          similarity,
          12,
        )
        expect(Array.from(scorePairs([left], [right], { scorer }))[0]).toBeCloseTo(
          similarity,
          12,
        )
        expect(scoreMatrix([left], [right], { scorer }).toArray()[0]?.[0]).toBeCloseTo(
          similarity,
          12,
        )
      },
    )
  })

  describe(`${name}'s direct call takes the default gram size`, () => {
    test.each(cases.filter((entry) => entry.gramSize === 2))(
      'case %#',
      ({ left, right, similarity }) => {
        expect(suite.similarity(left, right)).toBeCloseTo(similarity, 12)
        expect(suite.distance(left, right)).toBeCloseTo(1 - similarity, 12)
        expect(suite.normalizedSimilarity(left, right)).toBeCloseTo(similarity, 12)
        expect(suite.normalizedDistance(left, right)).toBeCloseTo(1 - similarity, 12)
      },
    )
  })
}

describe('the two families are not the same metric', () => {
  test('a repeated gram separates them', () => {
    // Dice weights a shared gram by the smaller multiplicity; cosine weights it
    // by the product, and normalizes by the vector lengths rather than the
    // counts. `textdistance.Cosine` answers 0.75 here for a third formula
    // again — intersection count over the geometric mean of the totals.
    const left = ['ab', 'ab', 'ab', 'bc']
    const right = ['ab', 'ab', 'bc', 'bc']
    expect(createScorer(dice.similarity, { gramSize: 1 }).score(left, right)).toBeCloseTo(
      0.75,
      12,
    )
    expect(
      createScorer(cosine.similarity, { gramSize: 1 }).score(left, right),
    ).toBeCloseTo(8 / Math.sqrt(10 * 8), 12)
  })
})
