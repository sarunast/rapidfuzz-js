import { describe, expect, it } from 'vitest'

import {
  distance as damerauDistance,
  similarity as damerauSimilarity,
} from '../../src/algorithms/damerauLevenshtein/index.js'
import {
  distance as hammingDistance,
  similarity as hammingSimilarity,
} from '../../src/algorithms/hamming/index.js'
import {
  distance as indelDistance,
  similarity as indelSimilarity,
} from '../../src/algorithms/indel/index.js'
import { similarity as jaroSimilarity } from '../../src/algorithms/jaro/index.js'
import { similarity as jaroWinklerSimilarity } from '../../src/algorithms/jaroWinkler/index.js'
import {
  distance as lcsDistance,
  similarity as lcsSimilarity,
} from '../../src/algorithms/lcs/index.js'
import {
  distance as levenshteinDistance,
  similarity as levenshteinSimilarity,
} from '../../src/algorithms/levenshtein/index.js'
import {
  distance as osaDistance,
  similarity as osaSimilarity,
} from '../../src/algorithms/osa/index.js'
import {
  distance as postfixDistance,
  similarity as postfixSimilarity,
} from '../../src/algorithms/postfix/index.js'
import {
  distance as prefixDistance,
  similarity as prefixSimilarity,
} from '../../src/algorithms/prefix/index.js'
import { createScorer, type Scorer } from '../../src/core/scorer.js'
import type { Direction } from '../../src/core/types.js'
import { createMatcher } from '../../src/search/index.js'

type ScorerFactory = () => Scorer<Direction>

const FAMILIES: ReadonlyArray<readonly [string, ScorerFactory]> = [
  ['Damerau-Levenshtein distance', () => createScorer(damerauDistance)],
  ['Damerau-Levenshtein similarity', () => createScorer(damerauSimilarity)],
  ['Hamming distance', () => createScorer(hammingDistance)],
  ['Hamming similarity', () => createScorer(hammingSimilarity)],
  ['Indel distance', () => createScorer(indelDistance)],
  ['Indel similarity', () => createScorer(indelSimilarity)],
  ['Jaro similarity', () => createScorer(jaroSimilarity)],
  ['Jaro-Winkler similarity', () => createScorer(jaroWinklerSimilarity)],
  ['LCS distance', () => createScorer(lcsDistance)],
  ['LCS similarity', () => createScorer(lcsSimilarity)],
  ['Levenshtein distance', () => createScorer(levenshteinDistance)],
  ['Levenshtein similarity', () => createScorer(levenshteinSimilarity)],
  ['OSA distance', () => createScorer(osaDistance)],
  ['OSA similarity', () => createScorer(osaSimilarity)],
  ['Postfix distance', () => createScorer(postfixDistance)],
  ['Postfix similarity', () => createScorer(postfixSimilarity)],
  ['Prefix distance', () => createScorer(prefixDistance)],
  ['Prefix similarity', () => createScorer(prefixSimilarity)],
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
    const similarity = createScorer(levenshteinSimilarity)
    const distance = createScorer(levenshteinDistance)
    const similarityMatcher = createMatcher(['axc'], { scorer: similarity })
    const distanceMatcher = createMatcher(['axc'], { scorer: distance })
    expect(similarityMatcher.best('abc', { threshold: 2 })).toBeUndefined()
    expect(similarityMatcher.best('abc', { threshold: 0 })?.score).toBeCloseTo(2 / 3)
    expect(distanceMatcher.best('abc', { threshold: -1 })).toBeUndefined()

    const osaDistanceScorer = createScorer(osaDistance)
    const osaSimilarityScorer = createScorer(osaSimilarity)
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
})
