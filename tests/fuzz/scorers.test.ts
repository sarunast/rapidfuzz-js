// Ported from RapidFuzz tests/test_fuzz.py
import { describe, expect, it } from 'vitest'

import { normalizeText as defaultProcess } from '../../src/core/normalize.js'
import { createScorer } from '../../src/core/scorer.js'
import * as publicFuzz from '../../src/fuzz/index.js'
import { partialRatio, partialRatioAlignment } from '../../src/fuzz/partial.js'
import { partialTokenRatio } from '../../src/fuzz/partialToken.js'
import { partialTokenSetRatio } from '../../src/fuzz/partialTokenSet.js'
import { partialTokenSortRatio } from '../../src/fuzz/partialTokenSort.js'
import { ratio } from '../../src/fuzz/similarity.js'
import { tokenRatio } from '../../src/fuzz/token.js'
import { tokenSetRatio } from '../../src/fuzz/tokenSet.js'
import { tokenSortRatio } from '../../src/fuzz/tokenSort.js'
import type { FuzzInput, FuzzOptions } from '../../src/fuzz/types.js'
import { wRatio } from '../../src/fuzz/weighted.js'
import { callUntyped } from '../support/common.js'

/**
 * Upstream wraps every call in `symmetric_scorer_tester`, which asserts the
 * score is the same with the arguments swapped. Every scorer below goes through
 * this wrapper for the same reason.
 */
type Scorer = (s1: FuzzInput, s2: FuzzInput, options?: FuzzOptions) => number

function symmetric(scorer: Scorer): Scorer {
  return (s1, s2, options) => {
    const score = scorer(s1, s2, options)
    expect(scorer(s2, s1, options)).toBeCloseTo(score, 9)
    return score
  }
}

const fuzz = {
  ratio: symmetric(ratio),
  partialRatio: symmetric(partialRatio),
  tokenSortRatio: symmetric(tokenSortRatio),
  tokenSetRatio: symmetric(tokenSetRatio),
  tokenRatio: symmetric(tokenRatio),
  partialTokenSortRatio: symmetric(partialTokenSortRatio),
  partialTokenSetRatio: symmetric(partialTokenSetRatio),
  partialTokenRatio: symmetric(partialTokenRatio),
  wRatio: symmetric(wRatio),
}

const SCORERS: ReadonlyArray<readonly [string, Scorer]> = Object.entries(fuzz)

it('is case sensitive without normalization', () => {
  expect(fuzz.ratio('new york mets', 'new york mets')).toBe(100)
  expect(fuzz.ratio('new york mets', 'new YORK mets')).not.toBe(100)
})

it('scores a substring as a perfect partial ratio', () => {
  expect(fuzz.partialRatio('new york mets', 'the wonderful new york mets')).toBe(100)
})

it('scores identical strings as a perfect token sort ratio', () => {
  expect(fuzz.tokenSortRatio('new york mets', 'new york mets')).toBe(100)
})

it('scores reordered tokens as a perfect partial token sort ratio', () => {
  expect(fuzz.partialTokenSortRatio('new york mets', 'new york mets')).toBe(100)
  expect(
    fuzz.partialTokenSortRatio(
      'new york mets vs atlanta braves',
      'atlanta braves vs new york mets',
    ),
  ).toBe(100)
})

it('scores reordered and subset tokens as a perfect token set ratio', () => {
  expect(
    fuzz.tokenSetRatio(
      'new york mets vs atlanta braves',
      'atlanta braves vs new york mets',
    ),
  ).toBe(100)
  expect(fuzz.tokenSetRatio('js', 'vue js')).toBe(100)
})

it('scores reordered tokens as a perfect partial token set ratio', () => {
  expect(
    fuzz.partialTokenSetRatio(
      'new york mets vs atlanta braves',
      'atlanta braves vs new york mets',
    ),
  ).toBe(100)
})

it('scores an equal WRatio as 100', () => {
  expect(
    fuzz.wRatio(defaultProcess('new york mets'), defaultProcess('new york mets')),
  ).toBe(100)
})

it('accepts text normalized before scoring', () => {
  expect(
    fuzz.wRatio(defaultProcess('new york mets'), defaultProcess('new YORK mets')),
  ).toBe(100)
})

it('scales a WRatio partial match by 0.9', () => {
  expect(fuzz.wRatio('new york mets', 'the wonderful new york mets')).toBe(90)
})

it('scales a WRatio misordered full match by 0.95', () => {
  expect(
    fuzz.wRatio('new york mets vs atlanta braves', 'atlanta braves vs new york mets'),
  ).toBe(95)
})

it('scores an identical WRatio as 100', () => {
  expect(fuzz.wRatio('new york mets', 'new york mets')).toBe(100)
})

it('scales a long partial match by 0.9 (issue 452)', () => {
  expect(fuzz.wRatio('hello', 'hello' + 'abcde'.repeat(7))).toBeCloseTo(90, 6)
})

it('finds the optimal partial alignment (issue 76)', () => {
  expect(fuzz.partialRatio('physics 2 vid', 'study physics physics 2')).toBeCloseTo(
    81.81818,
    4,
  )
  expect(fuzz.partialRatio('physics 2 vid', 'study physics physics 2 video')).toBe(100)
})

it('finds the optimal partial alignment (issue 90)', () => {
  expect(fuzz.partialRatio('ax b', 'a b a c b')).toBeCloseTo(85.71428, 4)
})

it('handles a needle longer than 64 characters (issue 138)', () => {
  const str1 = 'a'.repeat(65)
  const str2 = 'a' + String.fromCharCode(256) + 'a'.repeat(63)
  expect(fuzz.partialRatio(str1, str2)).toBeCloseTo(99.22481, 4)
})

it('reports the alignment of the partial match', () => {
  const a = 'a certain string'
  const s = 'certain'

  expect(partialRatioAlignment(s, a)).toEqual({
    score: 100,
    srcStart: 0,
    srcEnd: s.length,
    destStart: 2,
    destEnd: 2 + s.length,
  })
  expect(partialRatioAlignment(a, s)).toEqual({
    score: 100,
    srcStart: 2,
    srcEnd: 2 + s.length,
    destStart: 0,
    destEnd: s.length,
  })

  expect(partialRatioAlignment(null, 'test')).toBeNull()
  expect(partialRatioAlignment('test', null)).toBeNull()
  expect(partialRatioAlignment('test', 'tesx', { scoreCutoff: 90 })).toBeNull()
})

it('applies score_cutoff to WRatio (issue 196)', () => {
  expect(fuzz.wRatio('South Korea', 'North Korea')).toBeCloseTo(81.81818, 4)
  expect(fuzz.wRatio('South Korea', 'North Korea', { scoreCutoff: 85.4 })).toBe(0)
  expect(fuzz.wRatio('South Korea', 'North Korea', { scoreCutoff: 85.5 })).toBe(0)
})

it('does not return WRatio scores below score_cutoff when scaled above 100', () => {
  expect(fuzz.wRatio('b', ' b daadabbb')).toBe(60)
  expect(fuzz.wRatio('b', ' b daadabbb', { scoreCutoff: 59.9 })).toBe(60)
  expect(fuzz.wRatio('b', ' b daadabbb', { scoreCutoff: 61 })).toBe(0)
})

describe('a score_cutoff above 100 can never be reached', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer('abcd', 'abcd')).toBe(100)
      expect(scorer('abcd', 'abcd', { scoreCutoff: 100 })).toBe(100)
      expect(scorer('abcd', 'abcd', { scoreCutoff: 100.1 })).toBe(0)
    })
  }
})

// Two empty inputs are the one pair whose perfect score is awarded up front
// rather than reached through the scoring kernels, so they are also the one pair
// a cutoff above 100 could be handed back unchecked.
describe('a score_cutoff above 100 rejects two empty inputs too', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer('', '', { scoreCutoff: 100.1 })).toBe(0)
    })
  }

  // Whitespace-only inputs reach the partial scorers as empty token sequences,
  // which is how they get there without being empty themselves.
  it('including inputs that tokenise to nothing', () => {
    expect(partialTokenSortRatio('   ', '  ', { scoreCutoff: 100.1 })).toBe(0)
    expect(partialTokenSortRatio('   ', '  ')).toBe(100)
  })
})

it('partialRatioAlignment respects a score_cutoff above 100 on empty inputs', () => {
  expect(partialRatioAlignment('', '')).toEqual({
    score: 100,
    srcStart: 0,
    srcEnd: 0,
    destStart: 0,
    destEnd: 0,
  })
  expect(partialRatioAlignment('', '', { scoreCutoff: 100 })).not.toBeNull()
  expect(partialRatioAlignment('', '', { scoreCutoff: 100.1 })).toBeNull()
})

it('reports the alignment for a long partial match (issue 231)', () => {
  const str1 =
    'er merkantilismus förderte handle und verkehr mit teils marktkonformen, teils dirigistischen maßnahmen.'
  const str2 =
    'ils marktkonformen, teils dirigistischen maßnahmen. an der schwelle zum 19. jahrhundert entstand ein neu'

  const alignment = partialRatioAlignment(str1, str2)

  expect(alignment?.srcStart).toBe(0)
  expect(alignment?.srcEnd).toBe(103)
  expect(alignment?.destStart).toBe(0)
  expect(alignment?.destEnd).toBe(51)
})

it('treats two empty strings as a perfect match or as no match, per scorer', () => {
  // perfect match
  expect(fuzz.ratio('', '')).toBe(100)
  expect(fuzz.partialRatio('', '')).toBe(100)
  expect(fuzz.tokenSortRatio('', '')).toBe(100)
  expect(fuzz.partialTokenSortRatio('', '')).toBe(100)
  expect(fuzz.tokenRatio('', '')).toBe(100)
  expect(fuzz.partialTokenRatio('', '')).toBe(100)

  // no match
  expect(fuzz.wRatio('', '')).toBe(0)
  expect(fuzz.tokenSetRatio('', '')).toBe(0)
  expect(fuzz.partialTokenSetRatio('', '')).toBe(0)

  // no match when there are no words
  expect(fuzz.tokenSetRatio('    ', '    ')).toBe(0)
  expect(fuzz.partialTokenSetRatio('    ', '    ')).toBe(0)
})

// The split above is FuzzyWuzzy's, kept by RapidFuzz (issue 110), and the
// public metrics are what a caller reaches it through. Pinned here so that
// making the three of them answer 100 — which reads like fixing an
// inconsistency — fails as the compatibility break it would be.
it('carries the empty-input split through to the public scorers', () => {
  const scorers = [
    [publicFuzz.similarity, 100],
    [publicFuzz.partialSimilarity, 100],
    [publicFuzz.tokenSimilarity, 100],
    [publicFuzz.tokenSortSimilarity, 100],
    [publicFuzz.partialTokenSimilarity, 100],
    [publicFuzz.partialTokenSortSimilarity, 100],
    [publicFuzz.weightedSimilarity, 0],
    [publicFuzz.tokenSetSimilarity, 0],
    [publicFuzz.partialTokenSetSimilarity, 0],
  ] as const

  for (const [metric, expected] of scorers) {
    expect(createScorer(metric).score('', '')).toBe(expected)
  }
  for (const metric of [
    publicFuzz.tokenSetSimilarity,
    publicFuzz.partialTokenSetSimilarity,
  ]) {
    expect(createScorer(metric).score('    ', '    ')).toBe(0)
  }
  // Whitespace splits the three: `weightedSimilarity`'s zero is for an empty
  // side, and whitespace is not one — RapidFuzz answers 100 here too.
  expect(createScorer(publicFuzz.weightedSimilarity).score('    ', '    ')).toBe(100)
})

describe('invalid input throws', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(() => callUntyped(scorer, 1, 1)).toThrow(TypeError)
    })
  }
})

describe('arrays are treated like strings', () => {
  const text = 'the wonderful new york mets'

  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer(Array.from(text), Array.from(text))).toBe(100)
      expect(scorer(text, Array.from(text))).toBe(100)
      expect(scorer(Array.from(text), text)).toBe(100)
    })
  }
})

it('resolves mixed-token hash collisions using strict element equality', () => {
  const packedElements = ['aa', 'bb']
  const embeddedSeparator = ['aa\u0000string:bb']
  const firstObject = {}
  const secondObject = {}

  for (const scorer of [
    tokenSetRatio,
    tokenRatio,
    partialTokenSetRatio,
    partialTokenRatio,
  ]) {
    expect(scorer(packedElements, embeddedSeparator)).toBe(0)
    expect(scorer([firstObject], [secondObject])).toBe(0)
    expect(scorer([firstObject], [firstObject])).toBe(100)
  }
})

describe('byte arrays are treated like strings', () => {
  const text = 'the wonderful new york mets'
  const bytes = new TextEncoder().encode(text)

  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer(bytes, bytes)).toBe(100)
      expect(scorer(text, bytes)).toBe(100)
      expect(scorer(bytes, text)).toBe(100)
    })
  }
})

describe('only null and undefined are compatible missing inputs', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer('test', null)).toBe(0)
      expect(scorer(null, 'test')).toBe(0)
      expect(scorer('test', undefined)).toBe(0)
      expect(scorer(undefined, 'test')).toBe(0)
      expect(() => callUntyped(scorer, 'test', Number.NaN)).toThrow(TypeError)
      expect(() => callUntyped(scorer, Number.NaN, 'test')).toThrow(TypeError)
    })
  }
})

describe('simple unicode comparisons', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      expect(scorer('ÁÄ', 'ABCD')).toBe(0)
      expect(scorer('ÁÄ', 'ÁÄ')).toBe(100)
    })
  }
})

it('remains case sensitive without normalization', () => {
  expect(fuzz.ratio('new york mets', 'new YORK mets')).not.toBe(100)
})

describe('score_cutoff just below the score still returns it (issue 206)', () => {
  for (const [name, scorer] of SCORERS) {
    it(name, () => {
      const score1 = scorer('South Korea', 'North Korea')
      const score2 = scorer('South Korea', 'North Korea', {
        scoreCutoff: score1 - 0.0001,
      })
      expect(score2).toBe(score1)
    })
  }
})

it('finds the optimal partial alignment on a long repetitive input (issue 257)', () => {
  const s1 = 'aaaaaaaaaaaaaaaaaaaaaaaabacaaaaaaaabaaabaaaaaaaababbbbbbbbbbabbcb'
  const s2 = 'aaaaaaaaaaaaaaaaaaaaaaaababaaaaaaaabaaabaaaaaaaababbbbbbbbbbabbcb'

  expect(fuzz.partialRatio(s1, s2)).toBeCloseTo(98.46153846153847, 6)
  expect(fuzz.partialRatio(s2, s1)).toBeCloseTo(98.46153846153847, 6)
})
