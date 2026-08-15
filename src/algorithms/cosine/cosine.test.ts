import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'
import { bestMatch, createMatcher } from '#search/index.js'

import { Cosine } from '../../../testing/scorers.js'
import { callUntyped } from '../../../testing/untyped.js'
import { similarity as diceMetric } from '../dice/index.js'
import { dotProduct } from '../ngram/compare.js'
import { buildProfile } from '../ngram/profile.js'
import { cosineDistance, cosineSimilarity } from './implementation.js'
import { distance as cosineDistanceMetric, similarity as cosineMetric } from './index.js'

describe('frequency-vector cosine', () => {
  it('weights a shared gram by the product of its counts', () => {
    // `ab:3, bc:1` against `ab:2, bc:2`: dot 8, norms sqrt(10) and sqrt(8).
    // Dice answers 0.75 here, and the intersection-count formula some libraries
    // ship as "cosine" answers 0.75 as well — both from different arithmetic.
    expect(
      Cosine.similarity(['ab', 'ab', 'ab', 'bc'], ['ab', 'ab', 'bc', 'bc'], {
        gramSize: 1,
      }),
    ).toBeCloseTo(8 / Math.sqrt(80), 12)
  })

  it('scores identical inputs exactly one', () => {
    // `Math.sqrt(s) * Math.sqrt(s)` is 3.0000000000000004 for s = 3, which
    // would leave these just short of 1.
    for (const text of ['a', 'ab', 'abc', 'banana', 'aaa', 'hello world']) {
      expect(cosineSimilarity(text, text), text).toBe(1)
      expect(cosineDistance(text, text), text).toBe(0)
    }
  })

  it('separates identical, disjoint and partly overlapping inputs', () => {
    expect(cosineSimilarity('abcdef', 'uvwxyz')).toBe(0)
    const partial = cosineSimilarity('the wonderful new york mets', 'new york mets')
    expect(partial).toBeGreaterThan(0)
    expect(partial).toBeLessThan(1)
  })
})

describe('sequences shorter than the gram size', () => {
  it('answers zero rather than NaN when only one side has grams', () => {
    // `‖A‖ = 0` with `‖B‖ > 0` is `0 / 0`. Dice's denominator saves it there;
    // cosine needs the branch.
    expect(cosineSimilarity('a', 'ab')).toBe(0)
    expect(cosineSimilarity('ab', 'a')).toBe(0)
    expect(cosineDistance('a', 'ab')).toBe(1)
  })

  it('falls back to equality when neither side has a gram', () => {
    expect(cosineSimilarity('', '')).toBe(1)
    expect(cosineSimilarity('a', 'a')).toBe(1)
    expect(cosineSimilarity('a', 'b')).toBe(0)
    expect(cosineSimilarity('a', '')).toBe(0)
    for (const gramSize of [1, 2, 3, 7]) {
      expect(Cosine.similarity('', '', { gramSize }), `gramSize ${gramSize}`).toBe(1)
      expect(Cosine.similarity('ab', 'ab', { gramSize }), `gramSize ${gramSize}`).toBe(1)
    }
  })

  it('gives a profile of nothing but unmatchable grams a norm of its own', () => {
    // Every `NaN`-bearing gram counts one toward the norm, so the denominator
    // is never zero and the score is 0 rather than NaN.
    expect(Cosine.similarity([Number.NaN, Number.NaN], [Number.NaN, Number.NaN])).toBe(0)
    expect(
      Cosine.similarity([Number.NaN, 1], [Number.NaN, 1], { gramSize: 1 }),
    ).toBeCloseTo(0.5, 12)
  })
})

describe('gram size configuration', () => {
  it('defaults to bigrams and reads the configured size', () => {
    expect(Cosine.similarity('night', 'nacht')).toBeCloseTo(0.25, 12)
    expect(Cosine.similarity('night', 'nacht', { gramSize: 1 })).toBeCloseTo(0.6, 12)
    expect(Cosine.similarity('night', 'nacht', { gramSize: 3 })).toBe(0)
  })

  it('rejects a gram size that is not a positive safe integer', () => {
    for (const gramSize of [0, -1, 1.5, Number.NaN, Infinity, 1e300]) {
      expect(
        () => createScorer(cosineMetric, { gramSize }),
        `gramSize ${gramSize}`,
      ).toThrow(RangeError)
    }
    expect(() => callUntyped(createScorer, cosineMetric, { gramSize: '2' })).toThrow(
      TypeError,
    )
  })

  it('treats an explicit default as no configuration at all', () => {
    const plain = createScorer(cosineMetric)
    const explicit = createScorer(cosineMetric, { gramSize: 2 })
    const deeper = createScorer(cosineMetric, { gramSize: 3 })
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

  it('does not accept a Dice profile', () => {
    // Two metrics, two identities. The profile is the same shape, which is
    // exactly why the brand rather than the structure decides — the compiler
    // refuses this outright, so the runtime check needs an untyped call.
    const options = {
      scorer: createScorer(cosineMetric),
      getPrepared: () => createScorer(diceMetric).prepareChoice('a'),
    }
    expect(() => Reflect.apply(bestMatch, undefined, ['a', ['a'], options])).toThrow(
      'prepared choice is incompatible with this scorer',
    )
  })
})

describe('thresholds', () => {
  it('returns the exact score whenever the candidate qualifies', () => {
    const scorer = createScorer(cosineMetric)
    const distance = createScorer(cosineDistanceMetric)
    const exact = cosineSimilarity('abcdef', 'abcxyz')

    expect(scorer.score('abcdef', 'abcxyz', { threshold: exact })).toBeCloseTo(exact, 12)
    expect(scorer.score('abcdef', 'abcxyz', { threshold: exact + 1e-9 })).toBeUndefined()
    expect(distance.score('abcdef', 'abcxyz', { threshold: 1 - exact })).toBeCloseTo(
      1 - exact,
      12,
    )
    expect(
      distance.score('abcdef', 'abcxyz', { threshold: 1 - exact - 1e-9 }),
    ).toBeUndefined()
    expect(
      createMatcher(['abcxyz'], { scorer }).best('abcdef', { threshold: exact + 1e-9 }),
    ).toBeUndefined()
  })

  it('applies a threshold to sequences that have no grams', () => {
    const scorer = createScorer(cosineMetric)
    expect(scorer.score('a', 'b', { threshold: 0.5 })).toBeUndefined()
    expect(scorer.score('a', 'a', { threshold: 0.5 })).toBe(1)
    expect(createMatcher(['b'], { scorer }).best('a', { threshold: 0.5 })).toBeUndefined()
    expect(createMatcher(['a'], { scorer }).best('a', { threshold: 0.5 })?.score).toBe(1)
  })
})

describe('the shared profile', () => {
  it('walks the dot product deeper than the call stack would allow', () => {
    const gramSize = 20_000
    const text = 'a'.repeat(gramSize) + 'b'
    expect(Cosine.similarity(text, text, { gramSize })).toBe(1)
  })

  it('multiplies counts where sharedFrequency takes the minimum', () => {
    const a = buildProfile('aaab', 1)
    const b = buildProfile('aab', 1)
    // a: {a:3, b:1}, b: {a:2, b:1} — 3*2 + 1*1.
    expect(dotProduct(a, b)).toBe(7)
    expect(a.squaredNorm).toBe(10)
    expect(b.squaredNorm).toBe(5)
    expect(dotProduct(a, buildProfile('xyz', 1))).toBe(0)
  })
})

describe('properties', () => {
  const sequences = fc.string({ maxLength: 24 })

  it('is symmetric, bounded, and complementary to its distance', () => {
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const scorer = createScorer(cosineMetric, { gramSize: n })
        const score = scorer.score(a, b)
        expect(score).toBeGreaterThanOrEqual(0)
        expect(score).toBeLessThanOrEqual(1)
        expect(scorer.score(b, a)).toBeCloseTo(score, 12)
        expect(scorer.score(a, a)).toBe(1)
        expect(
          createScorer(cosineDistanceMetric, { gramSize: n }).score(a, b),
        ).toBeCloseTo(1 - score, 12)
      }),
    )
  })

  it('scores a prepared choice the same as a direct call', () => {
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const scorer = createScorer(cosineMetric, { gramSize: n })
        const exact = scorer.score(a, b)
        const rows = [{ prepared: scorer.prepareChoice(b) }]
        expect(
          bestMatch(a, rows, { scorer, getPrepared: (row) => row.prepared })?.score,
        ).toBeCloseTo(exact, 12)
      }),
    )
  })
})
