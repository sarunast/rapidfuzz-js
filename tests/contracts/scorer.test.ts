import { describe, expect, test } from 'vitest'

import * as cosine from '../../src/algorithms/cosine/index.js'
import * as damerau from '../../src/algorithms/damerauLevenshtein/index.js'
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
import { scorerCompilation } from '../../src/core/scoring/scorer.js'
import { trustedKernelThreshold } from '../../src/core/scoring/threshold.js'
import * as fuzz from '../../src/fuzz/index.js'
import {
  bestMatch,
  createMatcher,
  createScorer,
  isMatch,
  normalizeText,
  scoreIfMatch,
} from '../../src/index.js'

describe('Metric and Scorer contracts', () => {
  test('trusted kernel thresholds normalize no-op distance bounds once', () => {
    expect(trustedKernelThreshold('distance', [0, 10], 10)).toBeNull()
    expect(trustedKernelThreshold('distance', [0, 10], 9)).toBe(9)
  })

  test('algorithm families keep their natural scales', () => {
    expect(fuzz.similarity('abc', 'axc')).toBeCloseTo(200 / 3)
    for (const metric of [
      levenshtein.normalizedSimilarity,
      indel.normalizedSimilarity,
      lcs.normalizedSimilarity,
      osa.normalizedSimilarity,
      damerau.normalizedSimilarity,
      hamming.normalizedSimilarity,
      prefix.normalizedSimilarity,
      postfix.normalizedSimilarity,
      jaro.similarity,
      jaroWinkler.similarity,
      dice.similarity,
      cosine.similarity,
    ]) {
      expect(metric('same', 'same')).toBe(1)
      expect(metric('a', 'b')).toBeGreaterThanOrEqual(0)
      expect(metric('a', 'b')).toBeLessThanOrEqual(1)
    }
    expect(levenshtein.distance('abc', 'axc')).toBe(1)
    expect(indel.distance('abc', 'axc')).toBe(2)
    expect(lcs.distance('abc', 'axc')).toBe(1)
    expect(osa.distance('ab', 'ba')).toBe(1)
    expect(damerau.distance('ab', 'ba')).toBe(1)
    expect(hamming.distance('abc', 'axc')).toBe(1)
    expect(levenshtein.similarity('abc', 'axc')).toBe(2)
    expect(indel.similarity('abc', 'axc')).toBe(4)
    for (const metric of [
      fuzz.partialSimilarity,
      fuzz.tokenSortSimilarity,
      fuzz.tokenSetSimilarity,
      fuzz.tokenSimilarity,
      fuzz.partialTokenSortSimilarity,
      fuzz.partialTokenSetSimilarity,
      fuzz.partialTokenSimilarity,
      fuzz.weightedSimilarity,
    ]) {
      expect(metric('new york mets', 'new york mets')).toBeGreaterThanOrEqual(0)
    }
    expect(fuzz.partialSimilarityAlignment('abc', 'zabc')?.score).toBe(100)
  })

  test('scorer metadata, configuration, freezing, and thresholds are scale aware', () => {
    const fuzzy = createScorer(fuzz.weightedSimilarity)
    const normalized = createScorer(levenshtein.normalizedSimilarity, {
      weights: { insertion: 1, deletion: 2, substitution: 1 },
    })
    const distance = createScorer(levenshtein.distance, {
      weights: [1, 2, 1],
    })
    expect(Object.isFrozen(fuzzy)).toBe(true)
    expect(fuzzy.bounds).toEqual([0, 100])
    expect(normalized.bounds).toEqual([0, 1])
    expect(normalized.symmetric).toBe(false)
    expect(distance.bounds).toEqual([0, Number.POSITIVE_INFINITY])
    expect(fuzzy.score('abc', 'axc', { threshold: 60 })).toBeCloseTo(200 / 3)
    expect(fuzzy.score('abc', 'axc', { threshold: 80 })).toBeUndefined()
    expect(normalized.score('abc', 'axc', { threshold: 0.6 })).toBeCloseTo(2 / 3)
    expect(normalized.score('abc', 'axc', { threshold: 0.8 })).toBeUndefined()
    expect(normalized.score('abc', 'axc', { threshold: -1 })).toBeCloseTo(2 / 3)
    expect(distance.score('abc', 'axc', { threshold: 1 })).toBe(1)
    expect(distance.score('abc', 'axc', { threshold: 0 })).toBeUndefined()
    for (const threshold of [Number.NaN, Infinity, -Infinity]) {
      expect(() => fuzzy.score('a', 'a', { threshold })).toThrow(RangeError)
    }
    expect(fuzzy.score('a', 'a', { threshold: 101 })).toBeUndefined()
  })

  test('compiled algorithm configuration owns nested mutable values', () => {
    const objectWeights = { insertion: 1, deletion: 1, substitution: 2 }
    const tupleWeights: [number, number, number] = [1, 1, 2]
    const fromObject = createScorer(levenshtein.distance, { weights: objectWeights })
    const fromTuple = createScorer(levenshtein.distance, { weights: tupleWeights })

    objectWeights.deletion = 100
    objectWeights.substitution = 100
    tupleWeights[1] = 100
    tupleWeights[2] = 100

    expect(fromObject.symmetric).toBe(true)
    expect(fromTuple.symmetric).toBe(true)
    expect(fromObject.score('a', 'b')).toBe(2)
    expect(fromTuple.score('a', 'b')).toBe(2)
  })

  test('missing and invalid inputs follow the scorer direction', () => {
    const compatible = createScorer(levenshtein.normalizedSimilarity)
    const throwing = createScorer(levenshtein.normalizedSimilarity, {
      missing: 'throw',
    })
    const distance = createScorer(levenshtein.distance)
    expect(compatible.score(null, 'abc')).toBe(0)
    expect(compatible.score(undefined, 'abc')).toBe(0)
    expect(() => throwing.score(null, 'abc')).toThrow(TypeError)
    expect(() => distance.score(null, 'abc')).toThrow(TypeError)
    for (const invalid of [1, true, Number.NaN, {}]) {
      expect(() => Reflect.apply(compatible.score, compatible, [invalid, 'abc'])).toThrow(
        TypeError,
      )
    }
    expect(() => Reflect.apply(fuzz.similarity, undefined, [Number.NaN, 'abc'])).toThrow(
      TypeError,
    )
    expect(compatible.score('', '')).toBe(1)
    expect(levenshtein.normalizedSimilarity(null, 'abc')).toBe(0)
  })

  test('custom results are always validated and custom bounds never prune execution', () => {
    let calls = 0
    const custom = createScorer(
      (a, b) => {
        calls++
        return a === b ? 5 : 2
      },
      { direction: 'similarity', bounds: [0, 5], symmetric: true },
    )
    expect(custom.score('a', 'b', { threshold: 9 })).toBeUndefined()
    expect(calls).toBe(1)
    expect(custom.score(null, 'b')).toBe(0)

    const invalid = createScorer(() => 6, {
      direction: 'similarity',
      bounds: [0, 5],
      symmetric: true,
    })
    expect(() => invalid.score('a', 'b', { threshold: 9 })).toThrow(RangeError)

    calls = 0
    expect(
      bestMatch('a', ['a', 'b'], {
        scorer: custom,
      }),
    ).toEqual({ item: 'a', key: 0, score: 5 })
    expect(calls).toBe(2)
    expect(createMatcher(['a', 'b'], { scorer: custom }).best('a')).toEqual({
      item: 'a',
      key: 0,
      score: 5,
    })
    expect(createMatcher(['a', 'b'], { scorer: custom }).search('a')).toEqual([
      { item: 'a', key: 0, score: 5 },
      { item: 'b', key: 1, score: 2 },
    ])

    for (const result of [Number.NaN, Infinity, -1]) {
      const broken = createScorer(() => result, {
        direction: 'similarity',
        bounds: [0, 1],
        symmetric: true,
      })
      expect(() => broken.score('a', 'b')).toThrow(RangeError)
    }
    expect(() =>
      Reflect.apply(createScorer, undefined, [() => 1, { direction: 'similarity' }]),
    ).toThrow(TypeError)
    expect(() => Reflect.apply(createScorer, undefined, [() => 1])).toThrow(TypeError)
    // Not an object at all: these must reach the intended refusal rather than
    // a `Reflect.get` complaint about a non-object target.
    for (const notAConfiguration of [null, 42, 'similarity', () => 1]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [() => 1, notAConfiguration]),
      ).toThrow('custom metrics require direction, bounds, and symmetric configuration')
    }
    // A custom distance scorer refuses `missing` as a built-in one does, rather
    // than accepting a word it would then ignore.
    for (const missing of ['compatible', 'throw']) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [
          () => 1,
          { direction: 'distance', bounds: [0, 1], symmetric: true, missing },
        ]),
      ).toThrow("unknown custom scorer configuration key 'missing'")
    }
    const customDistance = createScorer(() => 1, {
      direction: 'distance',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(() => customDistance.score(null, 'a')).toThrow(TypeError)
    // What a JavaScript caller can reach that a TypeScript one cannot. The
    // built-in test answers `false` for these rather than throwing on a symbol
    // lookup, so the refusal has to be made here, before a scorer exists.
    for (const notAMetric of [null, undefined, 42, 'levenshtein', {}]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [
          notAMetric,
          { direction: 'similarity', bounds: [0, 1], symmetric: true },
        ]),
      ).toThrow(TypeError)
    }
    // An inherited compile hook is not this package's metric: `builtInMetric`
    // installs it on the function itself, and a borrowed prototype should take
    // the custom path — which then refuses it for want of a configuration.
    const borrowed = Object.setPrototypeOf(() => 1, levenshtein.distance)
    expect(() => Reflect.apply(createScorer, undefined, [borrowed])).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'distance',
          bounds: 'no bounds',
          symmetric: true,
        },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'similarity',
          bounds: [0, 1],
          symmetric: 'yes',
        },
      ]),
    ).toThrow(TypeError)
    for (const bounds of [
      ['zero', 1],
      [0, 'one'],
      [Infinity, Infinity],
      [0, Number.NaN],
      [2, 1],
      [0, 1, 2],
    ]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [
          () => 1,
          {
            direction: 'similarity',
            bounds,
            symmetric: true,
          },
        ]),
      ).toThrow(RangeError)
    }
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'similarity',
          bounds: [10, 100],
          symmetric: true,
        },
      ]),
    ).toThrow(RangeError)
    const throwingBounds = createScorer(() => 10, {
      direction: 'similarity',
      bounds: [10, 100],
      symmetric: true,
      missing: 'throw',
    })
    expect(throwingBounds.score('a', 'a')).toBe(10)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'similarity',
          bounds: [0, 1],
          symmetric: true,
          missing: 'nope',
        },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        () => 1,
        {
          direction: 'similarity',
          bounds: [0, 1],
          symmetric: true,
          extra: true,
        },
      ]),
    ).toThrow(TypeError)
  })

  test('invalid built-in configuration is rejected at compilation', () => {
    expect(
      createScorer(levenshtein.similarity, { missing: 'compatible' }).score(null, 'a'),
    ).toBe(0)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        levenshtein.similarity,
        { missing: 'nope' },
      ]),
    ).toThrow(TypeError)
    for (const key of ['processor', 'scoreCutoff', 'scoreHint']) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [
          levenshtein.distance,
          { [key]: key === 'processor' ? (value: unknown) => value : 1 },
        ]),
      ).toThrow(TypeError)
    }
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        levenshtein.distance,
        { missing: 'compatible' },
      ]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [levenshtein.distance, { weights: [1, 2] }]),
    ).toThrow(TypeError)
    expect(() =>
      Reflect.apply(createScorer, undefined, [
        levenshtein.distance,
        {
          weights: [1, 'two', 3],
        },
      ]),
    ).toThrow(TypeError)
    // Not an object at all, the built-in mirror of the custom-metric case
    // above. `Object.keys` answers `[]` for a number and a boolean alike, so
    // each of these used to compile as if it were `{}`; a string instead
    // reached `Reflect.get` and failed with a complaint about our internals.
    for (const notAConfiguration of [null, 42, false, 'weights', () => 1]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [levenshtein.distance, notAConfiguration]),
      ).toThrow('metric configuration must be an object')
    }
    expect(createScorer(levenshtein.distance, undefined).score('abc', 'abd')).toBe(1)
  })

  test('built-in configuration values are settled at compilation', () => {
    // A widened value used to reach the kernels and mean two different things
    // there: Jaro-Winkler's direct path coerced `'0.2'` through its numeric
    // comparisons while preparation refused it, so `scorer.score` and a Matcher
    // built on the same scorer disagreed. `NaN` cleared both of their range
    // tests and poisoned every score instead.
    for (const metric of [jaroWinkler.similarity, jaroWinkler.distance]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [metric, { prefixWeight: '0.2' }]),
      ).toThrow(TypeError)
      expect(() =>
        Reflect.apply(createScorer, undefined, [metric, { prefixWeight: Number.NaN }]),
      ).toThrow(RangeError)
      expect(() =>
        Reflect.apply(createScorer, undefined, [metric, { prefixWeight: 1.5 }]),
      ).toThrow(RangeError)
    }
    const winkler = createScorer(jaroWinkler.similarity, { prefixWeight: 0.2 })
    expect(createMatcher(['abce'], { scorer: winkler }).best('abcd')?.score).toBeCloseTo(
      winkler.score('abcd', 'abce') ?? 0,
      12,
    )

    for (const metric of [
      hamming.distance,
      hamming.similarity,
      hamming.normalizedDistance,
      hamming.normalizedSimilarity,
    ]) {
      expect(() =>
        Reflect.apply(createScorer, undefined, [metric, { pad: 'yes' }]),
      ).toThrow(TypeError)
    }
    const strict = createScorer(hamming.distance, { pad: false })
    expect(strict.score('abc', 'abd')).toBe(1)
    expect(() => createMatcher(['abcd'], { scorer: strict }).best('abc')).toThrow(
      'Sequences are not the same length.',
    )
  })

  test('scoreIfMatch and isMatch use scorer thresholds', () => {
    const scorer = createScorer(levenshtein.normalizedSimilarity)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBeCloseTo(2 / 3)
    expect(scoreIfMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBeUndefined()
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.6 })).toBe(true)
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0.8 })).toBe(false)
    expect(isMatch(scorer, 'abc', 'axc', { threshold: 0 })).toBe(true)
    const custom = createScorer(() => 1, {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(isMatch(custom, 'a', 'b', { threshold: 0 })).toBe(true)
    const distance = createScorer(levenshtein.distance)
    expect(distance.score('a', 'a', { threshold: -1 })).toBeUndefined()
    // The distance half of the always-matches shortcut: a normalized distance
    // cannot exceed 1, so a threshold of 1 accepts every pair — including one
    // with nothing in common — while an invalid input still throws.
    const normalized = createScorer(levenshtein.normalizedDistance)
    expect(isMatch(normalized, 'abc', 'xyz', { threshold: 1 })).toBe(true)
    expect(isMatch(normalized, 'abc', 'xyz', { threshold: 0.5 })).toBe(false)
    expect(() => Reflect.apply(scorerCompilation, undefined, [{}])).toThrow(TypeError)
  })

  test('prepareChoice validates its argument and hides what it holds', () => {
    const scorer = createScorer(fuzz.tokenSetSimilarity)
    const handle = scorer.prepareChoice('new york mets')
    expect(Object.isFrozen(scorer)).toBe(true)
    expect(Object.keys(handle)).toEqual([])
    expect(JSON.stringify(handle)).toBe('{}')
    for (const invalid of [null, undefined, 1, true, Number.NaN, {}]) {
      expect(() => Reflect.apply(scorer.prepareChoice, scorer, [invalid])).toThrow(
        TypeError,
      )
    }
    expect(
      bestMatch('new york mets', [{ handle }], {
        scorer,
        getPrepared: (row) => row.handle,
      })?.score,
    ).toBe(100)
  })

  test('prepareChoice normalizes what it prepares, and records that it did', () => {
    const scorer = createScorer(fuzz.tokenSetSimilarity)
    const normalizing = { normalize: normalizeText }
    const handle = scorer.prepareChoice('New York Mets!', normalizing)
    // Normalized where it was prepared, so the handle holds the same text a
    // pre-normalized argument would have produced — and knows which function
    // produced it, which a caller doing that themselves could not record.
    expect(
      bestMatch('new york mets', [{ handle }], {
        scorer,
        getPrepared: (row) => row.handle,
        ...normalizing,
      })?.score,
    ).toBe(100)
    // Named but absent is the shape the type admits, and prepares nothing.
    expect(
      bestMatch('New York Mets!', [{ handle: scorer.prepareChoice('New York Mets!') }], {
        scorer,
        getPrepared: (row) => row.handle,
        normalize: undefined,
      })?.score,
    ).toBe(100)
    for (const invalid of [null, 1, 'nope', {}]) {
      expect(() =>
        Reflect.apply(scorer.prepareChoice, scorer, ['alpha', { normalize: invalid }]),
      ).toThrow('normalize must be a function')
    }
    expect(() => scorer.prepareChoice('alpha', { normalize: () => null })).toThrow(
      'normalize returned a missing value',
    )
    // `null` is not an absent option bag, and reading it with `?.` used to make
    // it one — an unnormalized handle from a call that asked for normalization.
    expect(() => Reflect.apply(scorer.prepareChoice, scorer, ['alpha', null])).toThrow(
      'prepareChoice options must be an object',
    )
    expect(() =>
      Reflect.apply(scorer.prepareChoice, scorer, [
        'alpha',
        { normalise: normalizeText },
      ]),
    ).toThrow("unknown prepareChoice option 'normalise'")
  })

  test('a handle owns its sequence, whatever the scorer prepares from', () => {
    // A handle outlives the call that made it, so mutating the sequence
    // afterwards must not reach through it. Built-in preparation converts a
    // string and a plain array-like, and used to keep a typed array by
    // reference; a custom metric is handed the sequence itself.
    const scorer = createScorer(levenshtein.distance)
    const typed = new Uint8Array([97, 98, 99])
    const array = [97, 98, 99]
    const custom = createScorer((a, b) => (String(a) === String(b) ? 1 : 0), {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    const mutable = ['a', 'b', 'c']
    const rows = [{ handle: scorer.prepareChoice(typed) }]
    const arrayRows = [{ handle: scorer.prepareChoice(array) }]
    const customRows = [{ handle: custom.prepareChoice(mutable) }]

    typed[0] = 120
    array[0] = 120
    mutable[0] = 'z'

    expect(
      bestMatch([97, 98, 99], rows, { scorer, getPrepared: (row) => row.handle })?.score,
    ).toBe(0)
    expect(
      bestMatch([97, 98, 99], arrayRows, { scorer, getPrepared: (row) => row.handle })
        ?.score,
    ).toBe(0)
    expect(
      bestMatch(['a', 'b', 'c'], customRows, {
        scorer: custom,
        getPrepared: (row) => row.handle,
      })?.score,
    ).toBe(1)
  })

  test('a custom metric prepares choices through its own protocol', () => {
    let calls = 0
    const custom = createScorer(
      (a, b) => {
        calls++
        return a === b ? 1 : 0
      },
      { direction: 'similarity', bounds: [0, 1], symmetric: true },
    )
    const rows = [{ handle: custom.prepareChoice('beta') }]
    expect(
      bestMatch('beta', rows, { scorer: custom, getPrepared: (row) => row.handle })
        ?.score,
    ).toBe(1)
    expect(calls).toBe(1)
    // A second custom scorer over the same function still owns its own
    // preparation: nothing about a caller's plain function is shareable.
    const other = createScorer(() => 1, {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
    })
    expect(() =>
      bestMatch('beta', rows, { scorer: other, getPrepared: (row) => row.handle }),
    ).toThrow(TypeError)
  })
})
