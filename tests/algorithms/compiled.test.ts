import { describe, expect, it } from 'vitest'

import {
  distance as cosineDistance,
  normalizedDistance as cosineNormalizedDistance,
  normalizedSimilarity as cosineNormalizedSimilarity,
  similarity as cosineSimilarity,
} from '../../src/algorithms/cosine/index.js'
import {
  distance as damerauDistance,
  normalizedDistance as damerauNormalizedDistance,
  normalizedSimilarity as damerauNormalizedSimilarity,
  similarity as damerauSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import {
  distance as diceDistance,
  normalizedDistance as diceNormalizedDistance,
  normalizedSimilarity as diceNormalizedSimilarity,
  similarity as diceSimilarity,
} from '../../src/algorithms/dice/index.js'
import {
  distance as hammingDistance,
  normalizedDistance as hammingNormalizedDistance,
  normalizedSimilarity as hammingNormalizedSimilarity,
  similarity as hammingSimilarity,
} from '../../src/algorithms/hamming/index.js'
import {
  distance as indelDistance,
  normalizedDistance as indelNormalizedDistance,
  normalizedSimilarity as indelNormalizedSimilarity,
  similarity as indelSimilarity,
} from '../../src/algorithms/indel/index.js'
import {
  distance as jaroDistance,
  normalizedDistance as jaroNormalizedDistance,
  normalizedSimilarity as jaroNormalizedSimilarity,
  similarity as jaroSimilarity,
} from '../../src/algorithms/jaro/index.js'
import {
  distance as jaroWinklerDistance,
  normalizedDistance as jaroWinklerNormalizedDistance,
  normalizedSimilarity as jaroWinklerNormalizedSimilarity,
  similarity as jaroWinklerSimilarity,
} from '../../src/algorithms/jaroWinkler/index.js'
import {
  distance as lcsDistance,
  normalizedDistance as lcsNormalizedDistance,
  normalizedSimilarity as lcsNormalizedSimilarity,
  similarity as lcsSimilarity,
} from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  normalizedDistance as levenshteinNormalizedDistance,
  normalizedSimilarity as levenshteinNormalizedSimilarity,
  similarity as levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/index.js'
import {
  distance as osaDistance,
  normalizedDistance as osaNormalizedDistance,
  normalizedSimilarity as osaNormalizedSimilarity,
  similarity as osaSimilarity,
} from '../../src/algorithms/osa/index.js'
import {
  distance as postfixDistance,
  normalizedDistance as postfixNormalizedDistance,
  normalizedSimilarity as postfixNormalizedSimilarity,
  similarity as postfixSimilarity,
} from '../../src/algorithms/postfix/index.js'
import {
  distance as prefixDistance,
  normalizedDistance as prefixNormalizedDistance,
  normalizedSimilarity as prefixNormalizedSimilarity,
  similarity as prefixSimilarity,
} from '../../src/algorithms/prefix/index.js'
import { scorePairs } from '../../src/batch/index.js'
import { createScorer, type Scorer } from '../../src/core/scorer.js'
import type { Direction } from '../../src/core/types.js'
import { createMatcher } from '../../src/search/index.js'

type ScorerFactory = () => Scorer<Direction>

const FAMILIES: ReadonlyArray<readonly [string, ScorerFactory]> = [
  ['Cosine distance', () => createScorer(cosineDistance)],
  ['Cosine similarity', () => createScorer(cosineSimilarity)],
  ['Cosine normalized distance', () => createScorer(cosineNormalizedDistance)],
  ['Cosine normalized similarity', () => createScorer(cosineNormalizedSimilarity)],
  ['Damerau-Levenshtein distance', () => createScorer(damerauDistance)],
  ['Damerau-Levenshtein similarity', () => createScorer(damerauSimilarity)],
  [
    'Damerau-Levenshtein normalized distance',
    () => createScorer(damerauNormalizedDistance),
  ],
  [
    'Damerau-Levenshtein normalized similarity',
    () => createScorer(damerauNormalizedSimilarity),
  ],
  ['Dice distance', () => createScorer(diceDistance)],
  ['Dice similarity', () => createScorer(diceSimilarity)],
  ['Dice normalized distance', () => createScorer(diceNormalizedDistance)],
  ['Dice normalized similarity', () => createScorer(diceNormalizedSimilarity)],
  ['Hamming distance', () => createScorer(hammingDistance)],
  ['Hamming similarity', () => createScorer(hammingSimilarity)],
  ['Hamming normalized distance', () => createScorer(hammingNormalizedDistance)],
  ['Hamming normalized similarity', () => createScorer(hammingNormalizedSimilarity)],
  ['Indel distance', () => createScorer(indelDistance)],
  ['Indel similarity', () => createScorer(indelSimilarity)],
  ['Indel normalized distance', () => createScorer(indelNormalizedDistance)],
  ['Indel normalized similarity', () => createScorer(indelNormalizedSimilarity)],
  ['Jaro distance', () => createScorer(jaroDistance)],
  ['Jaro similarity', () => createScorer(jaroSimilarity)],
  ['Jaro normalized distance', () => createScorer(jaroNormalizedDistance)],
  ['Jaro normalized similarity', () => createScorer(jaroNormalizedSimilarity)],
  ['Jaro-Winkler distance', () => createScorer(jaroWinklerDistance)],
  ['Jaro-Winkler similarity', () => createScorer(jaroWinklerSimilarity)],
  ['Jaro-Winkler normalized distance', () => createScorer(jaroWinklerNormalizedDistance)],
  [
    'Jaro-Winkler normalized similarity',
    () => createScorer(jaroWinklerNormalizedSimilarity),
  ],
  ['LCS distance', () => createScorer(lcsDistance)],
  ['LCS similarity', () => createScorer(lcsSimilarity)],
  ['LCS normalized distance', () => createScorer(lcsNormalizedDistance)],
  ['LCS normalized similarity', () => createScorer(lcsNormalizedSimilarity)],
  ['Levenshtein distance', () => createScorer(levenshteinDistance)],
  ['Levenshtein similarity', () => createScorer(levenshteinSimilarity)],
  ['Levenshtein normalized distance', () => createScorer(levenshteinNormalizedDistance)],
  [
    'Levenshtein normalized similarity',
    () => createScorer(levenshteinNormalizedSimilarity),
  ],
  ['OSA distance', () => createScorer(osaDistance)],
  ['OSA similarity', () => createScorer(osaSimilarity)],
  ['OSA normalized distance', () => createScorer(osaNormalizedDistance)],
  ['OSA normalized similarity', () => createScorer(osaNormalizedSimilarity)],
  ['Postfix distance', () => createScorer(postfixDistance)],
  ['Postfix similarity', () => createScorer(postfixSimilarity)],
  ['Postfix normalized distance', () => createScorer(postfixNormalizedDistance)],
  ['Postfix normalized similarity', () => createScorer(postfixNormalizedSimilarity)],
  ['Prefix distance', () => createScorer(prefixDistance)],
  ['Prefix similarity', () => createScorer(prefixSimilarity)],
  ['Prefix normalized distance', () => createScorer(prefixNormalizedDistance)],
  ['Prefix normalized similarity', () => createScorer(prefixNormalizedSimilarity)],
]

describe('compiled built-in metrics', () => {
  it.each(FAMILIES)('%s agrees across direct and prepared execution', (_name, make) => {
    const scorer = make()
    const query = 'the quick brown fox jumps over the lazy dog'
    const choice = 'the quick brown fox leaps over the lazy dog!'
    const direct = scorer.score(query, choice)
    const matcher = createMatcher([choice], { scorer })

    expect(matcher.best(query)?.score).toBeCloseTo(direct, 12)

    const threshold =
      scorer.direction === 'similarity'
        ? Math.max(scorer.bounds[0], direct - Number.EPSILON)
        : direct
    expect(matcher.best(query, { threshold })?.score).toBeCloseTo(direct, 12)
    expect(
      Array.from(scorePairs([query], [choice], { scorer, threshold }))[0],
    ).toBeCloseTo(direct, 12)
  })

  it('retains and specializes algorithm configuration', () => {
    const weighted = createScorer(levenshteinDistance, {
      weights: { insertion: 1, deletion: 1, substitution: 2 },
    })
    const strictHamming = createScorer(hammingDistance, { pad: false })
    const winkler = createScorer(jaroWinklerSimilarity, { prefixWeight: 0.25 })

    expect(weighted.score('kitten', 'sitting')).toBe(5)
    expect(() => strictHamming.score('a', 'ab')).toThrow(
      'Sequences are not the same length.',
    )
    expect(winkler.score('milyarder', 'milyarderlik')).toBe(1)
    expect(winkler.score('abcdx', 'abcdy', { threshold: 0.9 })).toBe(1)
    expect(winkler.score('abcd', 'wxyz', { threshold: 0.9 })).toBeUndefined()
    expect(
      createMatcher(['wxyz'], { scorer: winkler }).best('abcd', { threshold: 0.9 }),
    ).toBeUndefined()
  })

  it('uses trusted prepared bounds without entering an invalid kernel cutoff', () => {
    const similarity = createScorer(levenshteinNormalizedSimilarity)
    const distance = createScorer(levenshteinDistance)
    const similarityMatcher = createMatcher(['axc'], { scorer: similarity })
    const distanceMatcher = createMatcher(['axc'], { scorer: distance })
    expect(similarityMatcher.best('abc', { threshold: 2 })).toBeUndefined()
    expect(similarityMatcher.best('abc', { threshold: 0 })?.score).toBeCloseTo(2 / 3)
    expect(distanceMatcher.best('abc', { threshold: -1 })).toBeUndefined()

    const osaDistanceScorer = createScorer(osaDistance)
    const osaSimilarityScorer = createScorer(osaNormalizedSimilarity)
    expect(
      createMatcher(['abcdef'], { scorer: osaDistanceScorer }).best('a', {
        threshold: 2,
      }),
    ).toBeUndefined()
    expect(
      createMatcher(['abcdef'], { scorer: osaSimilarityScorer }).best('a', {
        threshold: 0.8,
      }),
    ).toBeUndefined()
  })

  it('returns each built-in numeric cutoff sentinel in batch scoring', () => {
    expect(
      Array.from(
        scorePairs(['abc'], ['axc'], {
          scorer: createScorer(levenshteinSimilarity),
          threshold: 3,
        }),
      ),
    ).toEqual([0])
    expect(
      Array.from(
        scorePairs(['abc'], ['axc'], {
          scorer: createScorer(levenshteinNormalizedDistance),
          threshold: 0.2,
        }),
      ),
    ).toEqual([1])
  })
})
