import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { Dice } from '../../../testing/scorers.js'
import { callUntyped } from '../../../testing/untyped.js'
import { scoreMatrix } from '../../batch/scoreMatrix.js'
import { scorePairs } from '../../batch/scorePairs.js'
import { createScorer } from '../../core/scoring/scorer.js'
import { elementsEqual } from '../../core/sequence.js'
import { bestMatch, createMatcher, search, searchIter } from '../../search/index.js'
import { sharedFrequency } from '../shared/ngram/compare.js'
import { validGramSize } from '../shared/ngram/gramSize.js'
import { buildProfile, preparedProfile } from '../shared/ngram/profile.js'
import { diceDistance, diceSimilarity } from './implementation.js'
import { distance as diceDistanceMetric, similarity as diceMetric } from './index.js'

describe('multiset n-gram similarity', () => {
  it('counts a repeated gram as often as both sides carry it', () => {
    // min(3, 2) + min(1, 2) = 3 shared out of 4 + 4 grams. A set-based Dice
    // would say 2 * 2 / (2 + 2) = 1 here, which is the whole reason this one
    // keeps the multiplicities.
    expect(
      Dice.similarity(['ab', 'ab', 'ab', 'bc'], ['ab', 'ab', 'bc', 'bc'], {
        gramSize: 1,
      }),
    ).toBeCloseTo(0.75, 12)
    // Set-based implementations answer 0.857143 for this pair.
    expect(diceSimilarity('banana', 'bananas')).toBeCloseTo(0.909091, 6)
  })

  it('scores the documented worked example', () => {
    // `ni ig gh ht` against `na ac ch ht` shares only `ht`.
    expect(diceSimilarity('night', 'nacht')).toBeCloseTo(0.25, 12)
    expect(diceDistance('night', 'nacht')).toBeCloseTo(0.75, 12)
  })

  it('separates identical, disjoint and partly overlapping inputs', () => {
    expect(diceSimilarity('hello world', 'hello world')).toBe(1)
    expect(diceSimilarity('abcdef', 'uvwxyz')).toBe(0)
    const partial = diceSimilarity('the wonderful new york mets', 'new york mets')
    expect(partial).toBeGreaterThan(0)
    expect(partial).toBeLessThan(1)
  })

  it('adds no padding at the ends', () => {
    // `aba` and `bab` have the same bigram multiset, `{ab, ba}`. An
    // implementation that wraps each input in guards compares `%a ab ba a%`
    // against `%b ba ab b%` instead and answers 0.5.
    expect(diceSimilarity('aba', 'bab')).toBe(1)
  })
})

describe('sequences shorter than the gram size', () => {
  it('falls back to equality when neither side has a gram', () => {
    expect(diceSimilarity('', '')).toBe(1)
    expect(diceSimilarity('a', 'a')).toBe(1)
    expect(diceSimilarity('a', 'b')).toBe(0)
    expect(diceSimilarity('a', '')).toBe(0)
    expect(diceSimilarity('a', 'ab')).toBe(0)
    expect(diceDistance('a', 'a')).toBe(0)
    expect(diceDistance('a', 'b')).toBe(1)
  })

  it('applies the same rule at every gram size', () => {
    for (const gramSize of [1, 2, 3, 7]) {
      expect(Dice.similarity('', '', { gramSize }), `gramSize ${gramSize}`).toBe(1)
      expect(Dice.similarity('ab', 'ab', { gramSize }), `gramSize ${gramSize}`).toBe(1)
      expect(Dice.similarity('ab', 'ba', { gramSize })).toBe(gramSize === 1 ? 1 : 0)
    }
  })
})

describe('gram size configuration', () => {
  it('defaults to bigrams and reads the configured size', () => {
    expect(Dice.similarity('night', 'nacht')).toBe(diceSimilarity('night', 'nacht'))
    expect(Dice.similarity('night', 'nacht', { gramSize: 1 })).toBeCloseTo(0.6, 12)
    expect(Dice.similarity('night', 'nacht', { gramSize: 3 })).toBe(0)
    expect(Dice.similarity('night', 'nacht', { gramSize: 9 })).toBe(0)
  })

  it('rejects a gram size that is not a positive safe integer', () => {
    for (const gramSize of [0, -1, 1.5, Number.NaN, Infinity, 1e300]) {
      expect(
        () => createScorer(diceMetric, { gramSize }),
        `gramSize ${gramSize}`,
      ).toThrow(RangeError)
    }
    expect(() => callUntyped(createScorer, diceMetric, { gramSize: '2' })).toThrow(
      TypeError,
    )
    expect(() => callUntyped(createScorer, diceMetric, { grams: 2 })).toThrow(
      "unknown metric configuration key 'grams'",
    )
  })

  it('treats an explicit default as no configuration at all', () => {
    const plain = createScorer(diceMetric)
    const explicit = createScorer(diceMetric, { gramSize: 2 })
    const deeper = createScorer(diceMetric, { gramSize: 3 })
    const rows = [{ prepared: plain.prepareChoice('alphabet') }]

    expect(
      bestMatch('alphabet', rows, {
        scorer: explicit,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(1)
    expect(() =>
      bestMatch('alphabet', rows, {
        scorer: deeper,
        getPrepared: (row) => row.prepared,
      }),
    ).toThrow('prepared choice is incompatible with this scorer')
  })
})

describe('generic sequences', () => {
  it('keeps arbitrary elements apart without serializing them', () => {
    expect(Dice.similarity([1, 2, 3], [1, 2, 4], { gramSize: 1 })).toBeCloseTo(2 / 3, 12)
    expect(diceSimilarity(['foo', 'bar'], ['foo', 'baz'])).toBe(0)
    // A separator-joined key would make these two the same single gram.
    expect(Dice.similarity(['a,b', 'c'], ['a', 'b,c'], { gramSize: 1 })).toBe(0)
  })

  it('compares object elements by identity', () => {
    const token = {}
    const other = {}
    expect(Dice.similarity([token, 1], [token, 2], { gramSize: 1 })).toBeCloseTo(0.5, 12)
    expect(Dice.similarity([token, 1], [other, 1], { gramSize: 1 })).toBeCloseTo(0.5, 12)
    expect(Dice.similarity([token, 1], [token, 1], { gramSize: 1 })).toBe(1)
  })

  it('treats a gram holding NaN as unmatchable', () => {
    // `Map` keys match `NaN` to itself under SameValueZero; every element
    // comparison in this library uses `===`, which does not.
    expect(
      Dice.similarity([Number.NaN, 1], [Number.NaN, 1], { gramSize: 1 }),
    ).toBeCloseTo(0.5, 12)
    expect(diceSimilarity([Number.NaN, 1], [Number.NaN, 1])).toBe(0)
    // The unmatchable grams still count in the denominator: `abc` vs `abc` with
    // a NaN appended shares two of its three bigrams.
    expect(diceSimilarity(['a', 'b', 'c', Number.NaN], ['a', 'b', 'c'])).toBeCloseTo(
      (2 * 2) / (3 + 2),
      12,
    )
  })

  it('agrees between a string and its elements', () => {
    expect(diceSimilarity('abc', ['a', 'b', 'c'])).toBe(1)
    expect(diceSimilarity('😀a', ['😀', 'a'])).toBe(1)
  })
})

describe('Unicode', () => {
  it('reads astral characters as single elements', () => {
    expect(diceSimilarity('😀😀', '😀😀')).toBe(1)
    expect(diceSimilarity('😀a', '😀b')).toBe(0)
    expect(Dice.similarity('😀a😀', '😀b😀', { gramSize: 1 })).toBeCloseTo(2 / 3, 12)
    expect(diceSimilarity('é', 'é')).toBe(1)
  })

  it('bounds the search on code points, not UTF-16 units', () => {
    // Regression: counting `'😀'.length` as 2 gives 15 grams for the right
    // side instead of 10, an upper bound of 0.5, and a 0.6 threshold that
    // rejects a pair whose real similarity is 0.667.
    const left = 'abcdef'
    const right = 'abcdef😀😀😀😀😀'
    const exact = diceSimilarity(left, right)
    expect(exact).toBeCloseTo(2 / 3, 12)

    const scorer = createScorer(diceMetric)
    expect(scorer.score(left, right, { threshold: 0.6 })).toBeCloseTo(exact, 12)
    expect(
      createMatcher([right], { scorer }).best(left, { threshold: 0.6 })?.score,
    ).toBeCloseTo(exact, 12)
  })
})

describe('thresholds', () => {
  it('returns the exact score whenever the candidate qualifies', () => {
    const scorer = createScorer(diceMetric)
    const distance = createScorer(diceDistanceMetric)
    const exact = diceSimilarity('the wonderful new york mets', 'new york mets')

    for (const threshold of [0, 0.25, exact]) {
      expect(
        scorer.score('the wonderful new york mets', 'new york mets', { threshold }),
      ).toBeCloseTo(exact, 12)
    }
    expect(
      scorer.score('the wonderful new york mets', 'new york mets', {
        threshold: exact + 1e-9,
      }),
    ).toBeUndefined()
    expect(
      distance.score('the wonderful new york mets', 'new york mets', {
        threshold: 1 - exact,
      }),
    ).toBeCloseTo(1 - exact, 12)
    expect(
      distance.score('the wonderful new york mets', 'new york mets', {
        threshold: 1 - exact - 1e-9,
      }),
    ).toBeUndefined()
  })

  it('reports a rejected candidate at the far end of its bounds', () => {
    const similarity = prepareScorerOf(diceSimilarity)('abcdef', {})
    const distance = prepareScorerOf(diceDistance)('abcdef', {})
    expect(similarity('abcdef', null)).toBe(1)
    expect(similarity('abcdef', 0.5)).toBe(1)
    expect(similarity('uvwxyz', 0.5)).toBe(0)
    expect(distance('uvwxyz', null)).toBe(1)
    expect(distance('abcdef', 0.5)).toBe(0)
    expect(distance('uvwxyz', 0.5)).toBe(1)
  })

  it('rejects on the gram counts before either trie is built', () => {
    // A five-gram query cannot exceed 2 * 5 / (5 + 50) against a fifty-gram
    // choice, so nothing below that bound needs to be scored at all.
    const scorer = createScorer(diceMetric)
    const short = 'abcdef'
    const long = 'abcdef'.repeat(9)
    expect(scorer.score(short, long, { threshold: 0.8 })).toBeUndefined()
    expect(scorer.score(short, long, { threshold: 0.1 })).toBeGreaterThan(0.1)
    expect(
      createMatcher([long], { scorer }).best(short, { threshold: 0.8 }),
    ).toBeUndefined()
  })

  it('rejects on the score where the bound alone would not', () => {
    // Equal lengths put the bound at 1, so these reach the trie and are turned
    // down on the similarity itself.
    const scorer = createScorer(diceMetric)
    expect(scorer.score('abcdef', 'abcxyz')).toBeCloseTo(0.4, 12)
    expect(scorer.score('abcdef', 'abcxyz', { threshold: 0.5 })).toBeUndefined()
    expect(
      createMatcher(['abcxyz'], { scorer }).best('abcdef', { threshold: 0.5 }),
    ).toBeUndefined()
  })

  it('gives the walk up early without moving a qualifying score', () => {
    // The prepared kernel stops once the query's remaining frequency cannot
    // reach the cutoff and returns a count below it, so a threshold has to
    // agree with the full walk everywhere — the exact boundary included, which
    // is what the slack in `relaxedShared` is for.
    const query = 'abcdefabc'
    const candidates = ['abcdefabc', 'abcdefxyz', 'xyzabcdef', 'uvwxyzuvw']
    for (const gramSize of [1, 2, 3]) {
      const scorer = createScorer(diceMetric, { gramSize })
      for (const candidate of candidates) {
        const exact = scorer.score(query, candidate)
        const matcher = createMatcher([candidate], { scorer })
        const label = `gramSize ${gramSize}, ${candidate}`
        expect(matcher.best(query)?.score, label).toBeCloseTo(exact, 12)
        expect(matcher.best(query, { threshold: exact })?.score, label).toBeCloseTo(
          exact,
          12,
        )
        expect(matcher.best(query, { threshold: exact + 1e-9 }), label).toBeUndefined()
        // High enough that the walk gives up on the unrelated candidates part
        // way through rather than at the last group.
        const strict = matcher.best(query, { threshold: 0.9 })
        expect(strict === undefined || strict.score >= 0.9, label).toBe(true)
      }
    }
  })

  it('applies a threshold to sequences that have no grams', () => {
    const scorer = createScorer(diceMetric)
    expect(scorer.score('a', 'b', { threshold: 0.5 })).toBeUndefined()
    expect(scorer.score('a', 'a', { threshold: 0.5 })).toBe(1)
    expect(createMatcher(['b'], { scorer }).best('a', { threshold: 0.5 })).toBeUndefined()
    expect(createMatcher(['a'], { scorer }).best('a', { threshold: 0.5 })?.score).toBe(1)
  })
})

describe('every execution path agrees', () => {
  it('scores a configured metric the same way everywhere', () => {
    const scorer = createScorer(diceMetric, { gramSize: 3 })
    const query = 'the quick brown fox jumps over the lazy dog'
    const choice = 'the quick brown fox leaps over the lazy dog!'
    const exact = scorer.score(query, choice)

    expect(exact).toBeGreaterThan(0)
    expect(createMatcher([choice], { scorer }).best(query)?.score).toBeCloseTo(exact, 12)
    expect(bestMatch(query, [choice], { scorer })?.score).toBeCloseTo(exact, 12)
    expect(search(query, [choice], { scorer, limit: null })[0]?.score).toBeCloseTo(
      exact,
      12,
    )
    expect(Array.from(searchIter(query, [choice], { scorer }))[0]?.score).toBeCloseTo(
      exact,
      12,
    )
    expect(Array.from(scorePairs([query], [choice], { scorer }))[0]).toBeCloseTo(
      exact,
      12,
    )
    expect(scoreMatrix([query], [choice], { scorer }).toArray()[0]?.[0]).toBeCloseTo(
      exact,
      12,
    )

    const rows = [{ prepared: scorer.prepareChoice(choice) }]
    expect(
      bestMatch(query, rows, { scorer, getPrepared: (row) => row.prepared })?.score,
    ).toBeCloseTo(exact, 12)
  })

  it('takes the smaller trigram count whichever side holds it', () => {
    // The prepared trigram kernel walks the query's counts and compares each
    // against the choice's, so a repeat has to be checked from both directions.
    const scorer = createScorer(diceMetric, { gramSize: 3 })
    const repeated = 'abcabcabc'
    const once = 'abcxyz'
    for (const [query, choice] of [
      [once, repeated],
      [repeated, once],
    ]) {
      const rows = [{ prepared: scorer.prepareChoice(choice) }]
      expect(
        bestMatch(query, rows, { scorer, getPrepared: (row) => row.prepared })?.score,
        `${query} vs ${choice}`,
      ).toBeCloseTo(scorer.score(query, choice), 12)
    }
  })

  it('keeps a prepared choice usable after its source array changes', () => {
    const scorer = createScorer(diceMetric)
    const source = Array.from('alphabet')
    const prepared = scorer.prepareChoice(source)
    source[0] = 'z'
    expect(
      bestMatch('alphabet', [{ prepared }], {
        scorer,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(1)
  })
})

describe('the profile itself', () => {
  it('walks a trie deeper than the call stack would allow', () => {
    // `gramSize` is caller-supplied and equals the trie depth, so a recursive
    // traversal would put a stack overflow inside the range of valid inputs.
    const gramSize = 20_000
    const text = 'a'.repeat(gramSize) + 'b'
    expect(Dice.similarity(text, text, { gramSize })).toBe(1)
    expect(Dice.similarity(text, 'b' + 'a'.repeat(gramSize), { gramSize })).toBeCloseTo(
      0.5,
      12,
    )
  })

  it('refuses a prepared value it did not build', () => {
    expect(() => preparedProfile({})).toThrow('invalid prepared n-gram profile')
    expect(preparedProfile(buildProfile('abc', 2)).gramCount).toBe(2)
  })

  it('counts grams from the converted elements', () => {
    expect(buildProfile('😀😀😀', 2).gramCount).toBe(2)
    expect(buildProfile('ab', 3).gramCount).toBe(0)
    expect(sharedFrequency(buildProfile('abcabc', 2), buildProfile('abc', 2))).toBe(2)
  })

  it('compares elements with strict equality', () => {
    expect(elementsEqual([1, 2], [1, 2])).toBe(true)
    expect(elementsEqual([1, 2], [1, 2, 3])).toBe(false)
    expect(elementsEqual([1, 2], [1, 3])).toBe(false)
    expect(elementsEqual([Number.NaN], [Number.NaN])).toBe(false)
  })

  it('validates a gram size once, wherever it arrives from', () => {
    expect(validGramSize(undefined)).toBe(2)
    expect(validGramSize(null)).toBe(2)
    expect(validGramSize(5)).toBe(5)
    expect(() => validGramSize('5')).toThrow('gramSize must be a number')
    expect(() => validGramSize(0)).toThrow(
      'gramSize has to be a safe integer of at least 1',
    )
  })
})

describe('properties', () => {
  const sequences = fc.string({ maxLength: 24 })

  it('is symmetric and bounded', () => {
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const scorer = createScorer(diceMetric, { gramSize: n })
        const score = scorer.score(a, b)
        expect(score).toBeGreaterThanOrEqual(0)
        expect(score).toBeLessThanOrEqual(1)
        expect(scorer.score(b, a)).toBeCloseTo(score, 12)
        expect(scorer.score(a, a)).toBe(1)
      }),
    )
  })

  it('keeps distance the complement of similarity', () => {
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const similarity = createScorer(diceMetric, { gramSize: n }).score(a, b)
        const distance = createScorer(diceDistanceMetric, { gramSize: n }).score(a, b)
        expect(distance).toBeCloseTo(1 - similarity, 12)
      }),
    )
  })

  it('never rejects a candidate its bound should have admitted', () => {
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const scorer = createScorer(diceMetric, { gramSize: n })
        const exact = scorer.score(a, b)
        expect(scorer.score(a, b, { threshold: exact })).toBeCloseTo(exact, 12)
        expect(
          createMatcher([b], { scorer }).best(a, { threshold: exact })?.score,
        ).toBeCloseTo(exact, 12)
      }),
    )
  })
})
