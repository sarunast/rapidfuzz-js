import { describe, expect, it } from 'vitest'

import * as cosine from '../../src/algorithms/cosine/index.js'
import * as damerauLevenshtein from '../../src/algorithms/damerauLevenshtein/index.js'
import * as dice from '../../src/algorithms/dice/index.js'
import * as hamming from '../../src/algorithms/hamming/index.js'
import * as indel from '../../src/algorithms/indel/index.js'
import * as jaro from '../../src/algorithms/jaro/index.js'
import * as jaroWinkler from '../../src/algorithms/jaroWinkler/index.js'
import * as lcs from '../../src/algorithms/lcs/index.js'
import * as levenshtein from '../../src/algorithms/levenshtein/index.js'
import * as osa from '../../src/algorithms/osa/index.js'
import * as postfix from '../../src/algorithms/postfix/index.js'
import * as prefix from '../../src/algorithms/prefix/index.js'
import * as tversky from '../../src/algorithms/tversky/index.js'
import { createScorer, type Scorer } from '../../src/core/scoring/scorer.js'
import * as fuzz from '../../src/fuzz/index.js'
import { createMatcher } from '../../src/search/matcher/createMatcher.js'

const SCORERS: ReadonlyArray<readonly [string, Scorer]> = [
  ['cosine.distance', createScorer(cosine.distance)],
  ['cosine.similarity', createScorer(cosine.similarity)],
  ['damerauLevenshtein.distance', createScorer(damerauLevenshtein.distance)],
  ['damerauLevenshtein.similarity', createScorer(damerauLevenshtein.similarity)],
  [
    'damerauLevenshtein.normalizedDistance',
    createScorer(damerauLevenshtein.normalizedDistance),
  ],
  [
    'damerauLevenshtein.normalizedSimilarity',
    createScorer(damerauLevenshtein.normalizedSimilarity),
  ],
  ['dice.distance', createScorer(dice.distance)],
  ['dice.similarity', createScorer(dice.similarity)],
  ['hamming.normalizedDistance', createScorer(hamming.normalizedDistance, { pad: true })],
  [
    'hamming.normalizedSimilarity',
    createScorer(hamming.normalizedSimilarity, { pad: true }),
  ],
  ['indel.normalizedDistance', createScorer(indel.normalizedDistance)],
  ['indel.normalizedSimilarity', createScorer(indel.normalizedSimilarity)],
  ['jaro.distance', createScorer(jaro.distance)],
  ['jaro.similarity', createScorer(jaro.similarity)],
  ['jaroWinkler.distance', createScorer(jaroWinkler.distance)],
  ['lcs.normalizedDistance', createScorer(lcs.normalizedDistance)],
  ['lcs.normalizedSimilarity', createScorer(lcs.normalizedSimilarity)],
  ['levenshtein.normalizedDistance', createScorer(levenshtein.normalizedDistance)],
  ['levenshtein.normalizedSimilarity', createScorer(levenshtein.normalizedSimilarity)],
  [
    'levenshtein.normalizedSimilarity, weighted',
    createScorer(levenshtein.normalizedSimilarity, { weights: [1, 1, 2] }),
  ],
  [
    'levenshtein.normalizedSimilarity, fractional weights',
    createScorer(levenshtein.normalizedSimilarity, { weights: [0.3, 0.7, 1.1] }),
  ],
  ['osa.normalizedDistance', createScorer(osa.normalizedDistance)],
  ['osa.normalizedSimilarity', createScorer(osa.normalizedSimilarity)],
  ['postfix.normalizedSimilarity', createScorer(postfix.normalizedSimilarity)],
  ['prefix.normalizedSimilarity', createScorer(prefix.normalizedSimilarity)],
  ['tversky.distance', createScorer(tversky.distance)],
  ['tversky.similarity', createScorer(tversky.similarity)],
  [
    'tversky.distance, weighted',
    createScorer(tversky.distance, {
      gramSize: 1,
      elementWeights: new Map([['a', 3]]),
    }),
  ],
  [
    'tversky.distance, soft',
    createScorer(tversky.distance, {
      gramSize: 1,
      elementSimilarity: {
        scorer: createScorer(indel.normalizedSimilarity),
        threshold: 0.5,
      },
    }),
  ],
  ['fuzz.ratio', createScorer(fuzz.ratio)],
  ['fuzz.tokenSortRatio', createScorer(fuzz.tokenSortRatio)],
  ['fuzz.tokenSetRatio', createScorer(fuzz.tokenSetRatio)],
  ['fuzz.tokenRatio', createScorer(fuzz.tokenRatio)],
]

// Where RapidFuzz's Python and C++ implementations both return 0 with
// `score_cutoff` set to the pair's own score (verified against 3.14.5), the
// boundary follows upstream rather than this contract.
const UPSTREAM_REJECTS: ReadonlyArray<readonly [string, Scorer, string, string]> = [
  ['jaroWinkler.similarity', createScorer(jaroWinkler.similarity), 'cbeefc', 'cbabfc'],
  ['fuzz.partialRatio', createScorer(fuzz.partialRatio), 'bd cacd', 'ac dcacd'],
  ['fuzz.partialTokenRatio', createScorer(fuzz.partialTokenRatio), 'bd cacd', 'ac dcacd'],
  ['fuzz.weightedRatio', createScorer(fuzz.weightedRatio), 'c ada bbbe', 'be fa  c c'],
]

const PINNED: ReadonlyArray<readonly [string, string]> = [
  ['abcdxyzuvwq', 'abcdKLMNOPQ'],
  ['bddf', 'ddbfa'],
  ['cbeefc', 'cbabfc'],
  ['cccdcecccd', 'abcebfcbb'],
  ['cddbaabcb', 'ccaadcfbbfd'],
  ['fbbadef', 'fbb'],
  ['bcf', 'cbc'],
  ['beae', 'bdfeae'],
]

function randomPairs(count: number): Array<readonly [string, string]> {
  let seed = 0x2545f491
  const next = (bound: number): number => {
    seed ^= seed << 13
    seed ^= seed >>> 17
    seed ^= seed << 5
    return (seed >>> 0) % bound
  }
  const word = (): string => {
    let text = ''
    const length = 1 + next(12)
    for (let at = 0; at < length; at++) text += 'abc def'[next(7)]
    return text
  }
  return Array.from({ length: count }, () => [word(), word()] as const)
}

const PAIRS = [...PINNED, ...randomPairs(1500)]

describe('a threshold equal to the score admits the pair', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      for (const [a, b] of PAIRS) {
        const score = scorer.score(a, b)
        expect(scorer.score(a, b, { threshold: score }), `${name}(${a}, ${b})`).toBe(
          score,
        )
        expect(
          createMatcher([b], { scorer }).best(a, { threshold: score })?.score,
          `${name}(${a}, prepared ${b})`,
        ).toBe(score)
      }
    })
  }
})

describe('a threshold equal to the score follows upstream where upstream rejects it', () => {
  for (const [name, scorer, a, b] of UPSTREAM_REJECTS) {
    it(name, () => {
      const score = scorer.score(a, b)
      expect(
        scorer.score(a, b, { threshold: score }),
        `${name}(${a}, ${b})`,
      ).toBeUndefined()
    })
  }
})
