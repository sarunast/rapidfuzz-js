import fc from 'fast-check'
import { describe, expect, it } from 'vitest'

import { scoreMatrix } from '#batch/scoreMatrix.js'
import { scorePairs } from '#batch/scorePairs.js'
import { createScorer } from '#core/scoring/scorer.js'
import { bestMatch, createMatcher, search, searchIter } from '#search/index.js'

import { prepareScorerOf } from '../../../testing/prepareScorer.js'
import { Tversky } from '../../../testing/scorers.js'
import { callUntyped } from '../../../testing/untyped.js'
import { similarity as diceMetric } from '../dice/index.js'
import { tverskyDistance, tverskySimilarity } from './implementation.js'
import {
  distance as tverskyDistanceMetric,
  similarity as tverskyMetric,
} from './index.js'

// Positive finite doubles order like their bit patterns, so stepping the
// pattern is stepping one ulp.
const ULP_VIEW = new DataView(new ArrayBuffer(8))

function ulpAbove(value: number): number {
  ULP_VIEW.setFloat64(0, value)
  ULP_VIEW.setBigUint64(0, ULP_VIEW.getBigUint64(0) + 1n)
  return ULP_VIEW.getFloat64(0)
}

function ulpBelow(value: number): number {
  ULP_VIEW.setFloat64(0, value)
  ULP_VIEW.setBigUint64(0, ULP_VIEW.getBigUint64(0) - 1n)
  return ULP_VIEW.getFloat64(0)
}

describe('multiset n-gram similarity', () => {
  it('counts a repeated gram as often as both sides carry it', () => {
    // min(3, 2) + min(1, 2) = 3 shared against 1 first-only and 1 second-only
    // gram. A set-based Tversky would say 2 / (2 + 0) = 1 here, which is the
    // whole reason this one keeps the multiplicities.
    expect(
      Tversky.similarity(['ab', 'ab', 'ab', 'bc'], ['ab', 'ab', 'bc', 'bc'], {
        gramSize: 1,
      }),
    ).toBeCloseTo(0.75, 12)
  })

  it('scores the documented worked examples', () => {
    // `ni ig gh ht` against `na ac ch ht` shares only `ht`.
    expect(tverskySimilarity('night', 'nacht')).toBeCloseTo(0.25, 12)
    expect(tverskyDistance('night', 'nacht')).toBeCloseTo(0.75, 12)

    const containment = createScorer(tverskyMetric, { alpha: 1, beta: 0 })
    expect(containment.score('bana', 'banana')).toBe(1)
    expect(containment.score('banana', 'bana')).toBeCloseTo(0.6, 12)
  })

  it('separates identical, disjoint and partly overlapping inputs', () => {
    expect(tverskySimilarity('hello world', 'hello world')).toBe(1)
    expect(tverskySimilarity('abcdef', 'uvwxyz')).toBe(0)
    const partial = tverskySimilarity('the wonderful new york mets', 'new york mets')
    expect(partial).toBeGreaterThan(0)
    expect(partial).toBeLessThan(1)
  })

  it('adds no padding at the ends', () => {
    // `aba` and `bab` have the same bigram multiset, `{ab, ba}`.
    expect(tverskySimilarity('aba', 'bab')).toBe(1)
    expect(Tversky.similarity('aba', 'bab', { alpha: 1, beta: 0 })).toBe(1)
  })
})

describe('the three equivalences', () => {
  it('is exactly Dice at the default weights', () => {
    for (const [a, b] of [
      ['night', 'nacht'],
      ['banana', 'bananas'],
      ['the wonderful new york mets', 'new york mets'],
      ['abcdef', 'uvwxyz'],
      ['', ''],
    ]) {
      expect(tverskySimilarity(a, b), `${a} vs ${b}`).toBe(diceMetric(a, b))
    }
  })

  it('is exactly Dice at the default weights, on generated inputs', () => {
    const sequences = fc.string({ maxLength: 24 })
    fc.assert(
      fc.property(sequences, sequences, fc.integer({ min: 1, max: 4 }), (a, b, n) => {
        const tversky = createScorer(tverskyMetric, { gramSize: n })
        const dice = createScorer(diceMetric, { gramSize: n })
        const exact = dice.score(a, b)
        expect(tversky.score(a, b)).toBe(exact)
        expect(tversky.score(a, b, { threshold: exact })).toBe(exact)
        // The prepared kernels differ inside — Dice hands its walk a minimum
        // shared count, Tversky deliberately does not — so the equivalence has
        // to hold through a matcher too, thresholded at the boundary included.
        const viaTversky = createMatcher([b], { scorer: tversky })
        const viaDice = createMatcher([b], { scorer: dice })
        expect(viaTversky.best(a)?.score).toBe(viaDice.best(a)?.score)
        expect(viaTversky.best(a, { threshold: exact })?.score).toBe(
          viaDice.best(a, { threshold: exact })?.score,
        )
      }),
    )
  })

  it('is multiset Jaccard at alpha and beta 1', () => {
    // shared / (gramsA + gramsB − shared): 1 shared bigram of 4 and 4.
    expect(Tversky.similarity('night', 'nacht', { alpha: 1, beta: 1 })).toBeCloseTo(
      1 / 7,
      12,
    )
    expect(Tversky.similarity('banana', 'bananas', { alpha: 1, beta: 1 })).toBeCloseTo(
      5 / 6,
      12,
    )
  })

  it('is query containment at alpha 1 and beta 0', () => {
    const containment = createScorer(tverskyMetric, {
      gramSize: 1,
      alpha: 1,
      beta: 0,
    })
    expect(containment.score(['google', 'ag'], ['google', 'deepmind', 'ag'])).toBe(1)
    expect(containment.score(['google', 'deepmind', 'ag'], ['google', 'ag'])).toBeCloseTo(
      2 / 3,
      12,
    )
    expect(containment.score(['swisscom'], ['swisscomm'])).toBe(0)
  })
})

describe('asymmetry', () => {
  it('reports the configured symmetry on the compiled scorer', () => {
    expect(createScorer(tverskyMetric).symmetric).toBe(true)
    expect(createScorer(tverskyMetric, { alpha: 1, beta: 1 }).symmetric).toBe(true)
    expect(createScorer(tverskyMetric, { alpha: 1, beta: 0 }).symmetric).toBe(false)
    expect(createScorer(tverskyDistanceMetric, { alpha: 0.2, beta: 0.7 }).symmetric).toBe(
      false,
    )
  })

  it('scores each orientation with its own side of the weights', () => {
    const forward = createScorer(tverskyMetric, { alpha: 1, beta: 0.1 })
    expect(forward.score('bana', 'banana')).not.toBeCloseTo(
      forward.score('banana', 'bana'),
      12,
    )
  })

  it('keeps argument-and-weight swapping bit-identical', () => {
    // Two bigrams each, one shared, one unmatched on either side. With the
    // penalty terms summed separately the swap only commutes one addition;
    // folding them into the denominator left-to-right instead makes this pair
    // differ in the last bit at these weights.
    const forward = createScorer(tverskyMetric, { alpha: 0.5, beta: 10 })
    const swapped = createScorer(tverskyMetric, { alpha: 10, beta: 0.5 })
    expect(forward.score('abc', 'abd')).toBe(swapped.score('abd', 'abc'))
  })

  it('swapping the arguments equals swapping the weights', () => {
    const sequences = fc.string({ maxLength: 24 })
    const weights = fc.constantFrom(0, 0.1, 0.5, 1, 2, 10)
    fc.assert(
      fc.property(sequences, sequences, weights, weights, (a, b, alpha, beta) => {
        fc.pre(alpha !== 0 || beta !== 0)
        const forward = createScorer(tverskyMetric, { alpha, beta })
        const swapped = createScorer(tverskyMetric, { alpha: beta, beta: alpha })
        expect(forward.score(a, b)).toBe(swapped.score(b, a))
      }),
    )
  })

  it('prunes on the counts in the orientation the weights allow, and only there', () => {
    // With alpha 1 and beta 0 the candidate's extra grams cost nothing, so a
    // contained five-gram query still bounds to 1 against a fifty-three-gram
    // candidate — while the reversed orientation bounds to 5/53 and cannot
    // qualify. The threshold answers follow the orientation, which is what
    // this pins; that the rejection also skips the overlap walk is the
    // bound's job and is not observable from here.
    const containment = createScorer(tverskyMetric, { alpha: 1, beta: 0 })
    const short = 'abcdef'
    const long = 'abcdef'.repeat(9)
    expect(containment.score(short, long, { threshold: 0.99 })).toBe(1)
    expect(containment.score(long, short, { threshold: 0.99 })).toBeUndefined()
    const matcher = createMatcher([short], { scorer: containment })
    expect(matcher.best(long, { threshold: 0.99 })).toBeUndefined()
    expect(matcher.best(long, { threshold: 0.05 })?.score).toBeCloseTo(5 / 53, 12)
  })
})

describe('sequences shorter than the gram size', () => {
  it('falls back to equality when neither side has a gram', () => {
    expect(tverskySimilarity('', '')).toBe(1)
    expect(tverskySimilarity('a', 'a')).toBe(1)
    expect(tverskySimilarity('a', 'b')).toBe(0)
    expect(tverskySimilarity('a', '')).toBe(0)
    expect(tverskySimilarity('a', 'ab')).toBe(0)
    expect(tverskyDistance('a', 'a')).toBe(0)
    expect(tverskyDistance('a', 'b')).toBe(1)
  })

  it('applies the same rule whatever the weights say', () => {
    // The zero-gram fallback answers before the weights are consulted, so even
    // a forgiving containment configuration cannot rescue an unequal pair.
    expect(Tversky.similarity('a', 'ab', { alpha: 1, beta: 0 })).toBe(0)
    expect(Tversky.similarity('ab', 'ab', { gramSize: 7, alpha: 1, beta: 0 })).toBe(1)
  })
})

describe('configuration', () => {
  it('rejects weights that are not finite non-negative numbers', () => {
    for (const value of [-1, -0.5, Number.NaN, Infinity, -Infinity]) {
      expect(
        () => createScorer(tverskyMetric, { alpha: value }),
        `alpha ${value}`,
      ).toThrow(RangeError)
      expect(() => createScorer(tverskyMetric, { beta: value }), `beta ${value}`).toThrow(
        RangeError,
      )
    }
    expect(() => callUntyped(createScorer, tverskyMetric, { alpha: '1' })).toThrow(
      'alpha must be a number',
    )
    expect(() => callUntyped(createScorer, tverskyMetric, { beta: '1' })).toThrow(
      'beta must be a number',
    )
    expect(() => callUntyped(createScorer, tverskyMetric, { gamma: 1 })).toThrow(
      "unknown metric configuration key 'gamma'",
    )
  })

  it('rejects null weights rather than reading them as the default', () => {
    // Only an absent property defaults; `null` is a value, and a wrong one.
    expect(() => callUntyped(createScorer, tverskyMetric, { alpha: null })).toThrow(
      'alpha must be a number',
    )
    expect(() => callUntyped(createScorer, tverskyMetric, { beta: null })).toThrow(
      'beta must be a number',
    )
  })

  it('rejects both weights at zero, and allows either alone', () => {
    expect(() => createScorer(tverskyMetric, { alpha: 0, beta: 0 })).toThrow(
      'alpha and beta must not both be zero',
    )
    expect(() => tverskySimilarity('ab', 'cd', { alpha: 0, beta: 0 })).toThrow(RangeError)
    expect(Tversky.similarity('bana', 'banana', { alpha: 0, beta: 1 })).toBeCloseTo(
      0.6,
      12,
    )
    expect(Tversky.similarity('bana', 'banana', { alpha: 1, beta: 0 })).toBe(1)
  })

  it('rejects a gram size that is not a positive safe integer', () => {
    for (const gramSize of [0, -1, 1.5, Number.NaN, Infinity]) {
      expect(
        () => createScorer(tverskyMetric, { gramSize }),
        `gramSize ${gramSize}`,
      ).toThrow(RangeError)
    }
    expect(() => callUntyped(createScorer, tverskyMetric, { gramSize: '2' })).toThrow(
      TypeError,
    )
  })

  it('treats the explicit defaults as no configuration at all', () => {
    const plain = createScorer(tverskyMetric)
    const explicit = createScorer(tverskyMetric, {
      gramSize: 2,
      alpha: 0.5,
      beta: 0.5,
    })
    const weighted = createScorer(tverskyMetric, { alpha: 1, beta: 0 })
    const rows = [{ prepared: plain.prepareChoice('alphabet') }]

    expect(
      bestMatch('alphabet', rows, {
        scorer: explicit,
        getPrepared: (row) => row.prepared,
      })?.score,
    ).toBe(1)
    expect(() =>
      bestMatch('alphabet', rows, {
        scorer: weighted,
        getPrepared: (row) => row.prepared,
      }),
    ).toThrow('prepared choice is incompatible with this scorer')
  })

  it('does not accept a Dice profile', () => {
    // Two metrics, two identities. The profile is the same shape, which is
    // exactly why the brand rather than the structure decides — the compiler
    // refuses this outright, so the runtime check needs an untyped call.
    const options = {
      scorer: createScorer(tverskyMetric),
      getPrepared: () => createScorer(diceMetric).prepareChoice('a'),
    }
    expect(() => Reflect.apply(bestMatch, undefined, ['a', ['a'], options])).toThrow(
      'prepared choice is incompatible with this scorer',
    )
  })

  it('keeps huge finite weights representable by scaling, inside the bounds', () => {
    // `alpha * firstOnly` alone would overflow to Infinity and flush a
    // representable score to 0; the scaled arithmetic keeps the true value.
    const extreme = createScorer(tverskyMetric, {
      alpha: Number.MAX_VALUE,
      beta: Number.MAX_VALUE,
    })
    const tiny = extreme.score('abcdef', 'abcxyz')
    expect(tiny).toBeGreaterThan(0)
    expect(tiny).toBeCloseTo(2 / Number.MAX_VALUE / 6, 312)
    expect(extreme.score('abcdef', 'abcdef')).toBe(1)
    const oneSided = createScorer(tverskyMetric, { alpha: Number.MAX_VALUE, beta: 0 })
    expect(oneSided.score('bana', 'banana')).toBe(1)
    const reversed = oneSided.score('banana', 'bana')
    expect(reversed).toBeGreaterThan(0)
    expect(reversed).toBeCloseTo(3 / Number.MAX_VALUE / 2, 312)
  })
})

describe('generic sequences', () => {
  it('keeps arbitrary elements apart without serializing them', () => {
    expect(Tversky.similarity([1, 2, 3], [1, 2, 4], { gramSize: 1 })).toBeCloseTo(
      2 / 3,
      12,
    )
    expect(tverskySimilarity(['foo', 'bar'], ['foo', 'baz'])).toBe(0)
    expect(Tversky.similarity(['a,b', 'c'], ['a', 'b,c'], { gramSize: 1 })).toBe(0)
  })

  it('agrees between a string and its elements', () => {
    expect(tverskySimilarity('abc', ['a', 'b', 'c'])).toBe(1)
    expect(tverskySimilarity('😀a', ['😀', 'a'])).toBe(1)
  })

  it('reads astral characters as single elements', () => {
    expect(tverskySimilarity('😀😀', '😀😀')).toBe(1)
    expect(tverskySimilarity('😀a', '😀b')).toBe(0)
    expect(Tversky.similarity('😀a😀', '😀b😀', { gramSize: 1 })).toBeCloseTo(2 / 3, 12)
  })
})

describe('thresholds', () => {
  it('returns the exact score whenever the candidate qualifies', () => {
    const scorer = createScorer(tverskyMetric, { alpha: 1, beta: 0.25 })
    const distance = createScorer(tverskyDistanceMetric, { alpha: 1, beta: 0.25 })
    const exact = scorer.score('the wonderful new york mets', 'new york mets')

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
    const options = { alpha: 1, beta: 0.25 }
    const similarity = prepareScorerOf(tverskySimilarity)('abcdef', options)
    const distance = prepareScorerOf(tverskyDistance)('abcdef', options)
    expect(similarity('abcdef', null)).toBe(1)
    expect(similarity('abcdef', 0.5)).toBe(1)
    expect(similarity('uvwxyz', 0.5)).toBe(0)
    expect(distance('uvwxyz', null)).toBe(1)
    expect(distance('abcdef', 0.5)).toBe(0)
    expect(distance('uvwxyz', 0.5)).toBe(1)
  })

  it('rejects on the score where the bound alone would not', () => {
    // Equal lengths put the default bound at 1, so these reach the profiles
    // and are turned down on the similarity itself.
    const scorer = createScorer(tverskyMetric, { alpha: 1, beta: 1 })
    expect(scorer.score('abcdef', 'abcxyz')).toBeCloseTo(0.25, 12)
    expect(scorer.score('abcdef', 'abcxyz', { threshold: 0.5 })).toBeUndefined()
    expect(
      createMatcher(['abcxyz'], { scorer }).best('abcdef', { threshold: 0.5 }),
    ).toBeUndefined()
  })

  it('holds the boundary to within one ulp of the exact score', () => {
    // The dangerous failure of any threshold shortcut is rejecting a
    // candidate whose exact score qualifies. The exact score and its two ulp
    // neighbours sit right on that edge.
    for (const options of [
      { gramSize: 2, alpha: 1, beta: 0.25 },
      { gramSize: 1, alpha: 0.2, beta: 0.7 },
      { gramSize: 3, alpha: 2, beta: 10 },
    ]) {
      const scorer = createScorer(tverskyMetric, options)
      const query = 'the wonderful new york mets'
      const choice = 'new york mets and the dogs'
      const exact = scorer.score(query, choice)
      expect(exact).toBeGreaterThan(0)
      expect(exact).toBeLessThan(1)
      const prepared = prepareScorerOf(tverskySimilarity)(query, options)
      expect(prepared(choice, exact)).toBe(exact)
      expect(prepared(choice, ulpBelow(exact))).toBe(exact)
      expect(prepared(choice, ulpAbove(exact))).toBe(0)
    }
  })

  it('never inverts a threshold into a minimum shared count', () => {
    // The regression that keeps the prepared walk unbounded. At tiny weights
    // the score rounds up to the brink of 1 while a real-number inversion of
    // the threshold demands seven shared grams of a candidate that qualifies
    // with six — and the keys are chosen so the packed walk meets every
    // query-only gram first, which is where a bounded walk would give up and
    // under-count the match into a false rejection.
    const options = { gramSize: 1, alpha: 2 ** -54, beta: 2 ** -54 }
    const scorer = createScorer(tverskyMetric, options)
    const query = [0, 1, 2, 3, 1000, 1001, 1002, 1003, 1004, 1005]
    const choice = [
      1000, 1001, 1002, 1003, 1004, 1005, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117,
      2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126,
    ]
    const exact = scorer.score(query, choice)
    expect(exact).toBe(0.9999999999999999)
    const prepared = prepareScorerOf(tverskySimilarity)(query, options)
    expect(prepared(choice, exact)).toBe(exact)
    expect(
      createMatcher([choice], { scorer }).best(query, { threshold: exact })?.score,
    ).toBe(exact)
  })

  it('applies a threshold to sequences that have no grams', () => {
    const scorer = createScorer(tverskyMetric, { alpha: 1, beta: 0 })
    expect(scorer.score('a', 'b', { threshold: 0.5 })).toBeUndefined()
    expect(scorer.score('a', 'a', { threshold: 0.5 })).toBe(1)
    expect(createMatcher(['b'], { scorer }).best('a', { threshold: 0.5 })).toBeUndefined()
    expect(createMatcher(['a'], { scorer }).best('a', { threshold: 0.5 })?.score).toBe(1)
  })
})

describe('long inputs take the transient counter', () => {
  it('agrees with the profile walk past the counter gate', () => {
    // 599 and 601 bigrams put both sides past the 512-gram gate at the counter
    // gram sizes, and a gram size of 4 stays on the profile walk for contrast.
    const left = 'abcdef'.repeat(100)
    const right = 'abcdef'.repeat(100) + 'xy'
    for (const configuration of [
      { gramSize: 2, alpha: 1, beta: 0 },
      { gramSize: 3, alpha: 1, beta: 0.1 },
      { gramSize: 4, alpha: 1, beta: 0.1 },
    ]) {
      const scorer = createScorer(tverskyMetric, configuration)
      const viaMatcher = createMatcher([right], { scorer }).best(left)?.score
      expect(viaMatcher, `gramSize ${configuration.gramSize}`).toBeCloseTo(
        scorer.score(left, right),
        12,
      )
    }
  })

  it('falls back to profiles when the elements cannot be packed', () => {
    // Multi-character tokens have no packing radix, so the counter declines
    // past the gate and the profile walk answers instead.
    const shared = Array.from({ length: 600 }, (_, index) => `w${index}`)
    const disjoint = Array.from({ length: 600 }, (_, index) => `x${index}`)
    expect(tverskySimilarity(shared, [...shared])).toBe(1)
    expect(tverskySimilarity(shared, disjoint)).toBe(0)
  })
})

describe('every execution path agrees', () => {
  it('scores a configured metric the same way everywhere', () => {
    const scorer = createScorer(tverskyMetric, { gramSize: 3, alpha: 1, beta: 0.25 })
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

  it('keeps a prepared choice usable after its source array changes', () => {
    const scorer = createScorer(tverskyMetric, { alpha: 1, beta: 0 })
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

describe('properties', () => {
  const sequences = fc.string({ maxLength: 24 })
  const weights = fc.constantFrom(0, 0.1, 0.5, 1, 2, 10)

  it('is bounded and reflexive at every weight', () => {
    fc.assert(
      fc.property(
        sequences,
        sequences,
        fc.integer({ min: 1, max: 4 }),
        weights,
        weights,
        (a, b, n, alpha, beta) => {
          fc.pre(alpha !== 0 || beta !== 0)
          const scorer = createScorer(tverskyMetric, { gramSize: n, alpha, beta })
          const score = scorer.score(a, b)
          expect(score).toBeGreaterThanOrEqual(0)
          expect(score).toBeLessThanOrEqual(1)
          expect(scorer.score(a, a)).toBe(1)
        },
      ),
    )
  })

  it('keeps distance the complement of similarity', () => {
    fc.assert(
      fc.property(sequences, sequences, weights, weights, (a, b, alpha, beta) => {
        fc.pre(alpha !== 0 || beta !== 0)
        const similarity = createScorer(tverskyMetric, { alpha, beta }).score(a, b)
        const distance = createScorer(tverskyDistanceMetric, { alpha, beta }).score(a, b)
        expect(distance).toBeCloseTo(1 - similarity, 12)
      }),
    )
  })

  it('never rejects a candidate its bound should have admitted', () => {
    fc.assert(
      fc.property(sequences, sequences, weights, weights, (a, b, alpha, beta) => {
        fc.pre(alpha !== 0 || beta !== 0)
        const scorer = createScorer(tverskyMetric, { alpha, beta })
        const exact = scorer.score(a, b)
        expect(scorer.score(a, b, { threshold: exact })).toBeCloseTo(exact, 12)
        expect(
          createMatcher([b], { scorer }).best(a, { threshold: exact })?.score,
        ).toBeCloseTo(exact, 12)
      }),
    )
  })

  it('answers a thresholded prepared score exactly as the direct one, at every weight magnitude', () => {
    // Weights spanning the double range prove the scaled arithmetic never
    // flips a verdict between the two paths.
    const magnitudes = fc
      .integer({ min: -160, max: 160 })
      .map((exponent) => 2 ** exponent)
    const anyWeight = fc.oneof(
      magnitudes,
      fc.constantFrom(0, 0.1, 0.5, 1, Number.MAX_VALUE),
    )
    const thresholds = fc.double({ min: 0, max: 1, noNaN: true })
    fc.assert(
      fc.property(
        sequences,
        sequences,
        anyWeight,
        anyWeight,
        thresholds,
        (a, b, alpha, beta, threshold) => {
          fc.pre(alpha !== 0 || beta !== 0)
          const options = { alpha, beta }
          const direct = createScorer(tverskyMetric, options).score(a, b)
          const prepared = prepareScorerOf(tverskySimilarity)(a, options)
          expect(prepared(b, threshold)).toBe(direct >= threshold ? direct : 0)
        },
      ),
    )
  })
})
