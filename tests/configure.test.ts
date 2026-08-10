// Not ported from RapidFuzz — upstream has no equivalent. Python forwards a
// scorer's own options as `**scorer_kwargs`, a bag handed to `process`
// alongside the scorer; `configure` bakes them into the scorer instead, so
// there is one thing to pass rather than two that have to agree.
//
// What needs pinning down is that a configured scorer is still a scorer in
// every sense `process` cares about: it carries flags, it keeps the prepared
// query path, and it stays a built-in so `extract` may tighten its cutoff. And
// that the flags it carries reflect the options, which is what replaced the
// `weights` sniff `cdist` used to do.
import { describe, expect, it } from 'vitest'

import { isBuiltInScorer, prepareScorerOf, scorerFlagsOf } from '../src/_common.js'
import { configure } from '../src/configure.js'
import { hammingDistance } from '../src/distance/hamming.js'
import { jaroWinklerSimilarity } from '../src/distance/jaroWinkler.js'
import {
  levenshteinDistance,
  levenshteinNormalizedDistance,
  type LevenshteinWeights,
} from '../src/distance/levenshtein.js'
import { ratio } from '../src/fuzz.js'
import { extractOne } from '../src/search.js'
import { defaultProcess } from '../src/utils.js'
import { matrixScores } from './matrix.js'

describe('baked options reach the scorer', () => {
  // `[5, 5, 3]` and not `[1, 1, 2]`: a substitution never costs more than a
  // deletion plus an insertion, so weights where it would are indistinguishable
  // from the indel path and prove nothing about the option arriving.
  it('applies them to a direct call', () => {
    const weighted = configure(levenshteinDistance, { weights: [5, 5, 3] })
    expect(weighted('abc', 'abd')).toBe(3)
    expect(levenshteinDistance('abc', 'abd')).toBe(1)
  })

  it('lets a per-call option win', () => {
    const weighted = configure(levenshteinDistance, { weights: [5, 5, 3] })
    expect(weighted('abc', 'abd', { weights: [1, 1, 1] })).toBe(1)
  })

  it('still honours the per-call bounds it cannot bake', () => {
    const weighted = configure(levenshteinDistance, { weights: [5, 5, 3] })
    expect(weighted('abc', 'abd', { scoreCutoff: 1 })).toBe(
      levenshteinDistance('abc', 'abd', { weights: [5, 5, 3], scoreCutoff: 1 }),
    )
  })

  it('rejects an option the scorer parses eagerly', () => {
    expect(() => configure(hammingDistance, { pad: false })('abc', 'ab')).toThrow(
      'Sequences are not the same length.',
    )
  })

  it('carries through process', () => {
    const scorer = configure(jaroWinklerSimilarity, { prefixWeight: 0.2 })
    expect(extractOne('martha', ['marhta'], { scorer })?.score).toBe(
      jaroWinklerSimilarity('martha', 'marhta', { prefixWeight: 0.2 }),
    )
  })
})

describe('flags follow the baked options', () => {
  // The replacement for `cdist` reading the literal key `weights` off a kwargs
  // bag. Levenshtein answers the question itself, because it owns the option.
  it('asymmetric costs make the scorer asymmetric', () => {
    expect(scorerFlagsOf(configure(levenshteinDistance, { weights: [1, 2, 1] }))).toEqual(
      {
        ...scorerFlagsOf(levenshteinDistance),
        symmetric: false,
      },
    )
  })

  it('equal insertion and deletion costs stay symmetric', () => {
    const sets: readonly LevenshteinWeights[] = [
      [1, 1, 2],
      [2, 2, 5],
    ]
    for (const weights of sets) {
      expect(scorerFlagsOf(configure(levenshteinDistance, { weights })).symmetric).toBe(
        true,
      )
    }
  })

  it('keeps the direction of the metric it wraps', () => {
    const distance = scorerFlagsOf(configure(levenshteinDistance, { weights: [1, 2, 1] }))
    const normalized = scorerFlagsOf(
      configure(levenshteinNormalizedDistance, { weights: [1, 2, 1] }),
    )
    expect(distance.optimalScore).toBe(scorerFlagsOf(levenshteinDistance).optimalScore)
    expect(normalized.worstScore).toBe(
      scorerFlagsOf(levenshteinNormalizedDistance).worstScore,
    )
  })

  it('stops a matrix mirroring a triangle it has not scored', () => {
    const queries = ['abc', 'ab']
    const scorer = configure(levenshteinDistance, { weights: [1, 2, 1] })
    expect(matrixScores(queries, queries, { scorer })).toEqual([
      [0, levenshteinDistance('abc', 'ab', { weights: [1, 2, 1] })],
      [levenshteinDistance('ab', 'abc', { weights: [1, 2, 1] }), 0],
    ])
  })

  it('leaves a scorer with no resolver on its static flags', () => {
    expect(scorerFlagsOf(configure(ratio, {}))).toEqual(scorerFlagsOf(ratio))
  })
})

describe('what a configured scorer is registered as', () => {
  it('keeps the prepared path and built-in identity', () => {
    const scorer = configure(levenshteinDistance, { weights: [1, 1, 2] })
    expect(prepareScorerOf(scorer)).not.toBeNull()
    expect(isBuiltInScorer(scorer)).toBe(true)
  })

  it('agrees with a direct call through the prepared path', () => {
    const weights: LevenshteinWeights = [3, 7, 5]
    const scorer = configure(levenshteinDistance, { weights })
    const queries = ['martha', 'kitten', '']
    const choices = ['marhta', 'sitting', 'abc']
    expect(matrixScores(queries, choices, { scorer })).toEqual(
      queries.map((q) => choices.map((c) => levenshteinDistance(q, c, { weights }))),
    )
  })

  // A baked processor has to reach the scorer, and the prepared path bypasses
  // it. Being a built-in is separate: the processor runs once per choice
  // however `extract` moves the cutoff, and never observes it.
  it('drops the prepared path for a baked processor but stays a built-in', () => {
    const scorer = configure(ratio, { processor: defaultProcess })
    expect(prepareScorerOf(scorer)).toBeNull()
    expect(isBuiltInScorer(scorer)).toBe(true)
    expect(scorer('NEW YORK METS', 'new york mets')).toBe(100)
    expect(extractOne('NEW YORK METS', ['new york mets'], { scorer })?.score).toBe(100)
  })

  // Whether a scorer has a prepared path is decided once, when it is
  // configured, and a per-call option cannot give it one back. Overriding the
  // baked processor per call therefore still scores correctly — it just scores
  // through the unprepared path. Pinned so that making preparation depend on
  // the call is recognised as the architecture change it would be.
  it('keeps the prepared path off when a call overrides the baked processor', () => {
    const scorer = configure(ratio, { processor: defaultProcess })
    expect(scorer(' A ', 'a', { processor: undefined })).toBe(
      ratio(' A ', 'a', { processor: undefined }),
    )
    expect(prepareScorerOf(scorer)).toBeNull()
  })

  it('scores a matrix correctly without the choice-hoisting hook', () => {
    const scorer = configure(ratio, { processor: defaultProcess })
    const queries = ['NEW YORK', 'Chicago']
    const choices = ['new york', 'chicago!', 'boston']
    expect(matrixScores(queries, choices, { scorer })).toEqual(
      queries.map((q) => choices.map((c) => ratio(q, c, { processor: defaultProcess }))),
    )
  })

  // Registering it would change how many times `process` calls it, and
  // upstream's call count for a custom scorer is observable.
  it('gives a third-party scorer flags only', () => {
    let calls = 0
    const custom = (): number => {
      calls++
      return 50
    }
    const scorer = configure(custom, {})
    expect(isBuiltInScorer(scorer)).toBe(false)
    expect(prepareScorerOf(scorer)).toBeNull()
    expect(scorerFlagsOf(scorer)).toEqual(scorerFlagsOf(custom))

    matrixScores(['a', 'b'], ['c', 'd'], { scorer })
    expect(calls).toBe(4)
  })
})

// Regression: `configure` copied the options object but not the values inside
// it, and the flags resolver runs exactly once. So baking a symmetric weighting
// and then mutating it left a scorer that scored asymmetrically while its
// recorded flags still let `scoreMatrix` mirror half the matrix — a wrong
// number, not a stale one.
describe('baked options are snapshotted', () => {
  it('ignores a later mutation of a named cost object', () => {
    const weights = { insertion: 1, deletion: 1, substitution: 1 }
    const scorer = configure(levenshteinDistance, { weights })

    weights.deletion = 2

    expect(scorerFlagsOf(scorer).symmetric).toBe(true)
    expect(scorer('abc', 'ab')).toBe(1)
    expect(matrixScores(['abc', 'ab'], ['abc', 'ab'], { scorer })).toEqual([
      [0, 1],
      [1, 0],
    ])
  })

  it('ignores a later mutation of a weights tuple', () => {
    // `LevenshteinWeights` is `readonly`, but a mutable tuple is assignable to
    // it — so this is reachable from TypeScript as well as from JavaScript.
    const weights: [number, number, number] = [1, 1, 1]
    const scorer = configure(levenshteinDistance, { weights })

    weights[1] = 2

    expect(scorerFlagsOf(scorer).symmetric).toBe(true)
    expect(scorer('abc', 'ab')).toBe(1)
  })

  it('keeps the snapshot through a nested configure', () => {
    const weights = { insertion: 1, deletion: 1, substitution: 1 }
    const once = configure(levenshteinDistance, { weights })
    const twice = configure(once, { processor: defaultProcess })

    weights.deletion = 5

    expect(scorerFlagsOf(twice).symmetric).toBe(true)
    expect(twice('ABC', 'ab')).toBe(1)
  })
})

// `ScorerConfig` omits both, so only a JavaScript caller gets here — and the
// consequence is not a silently ignored option but a scorer that disagrees with
// itself, since `scoreMatrix` and `extract*` supply their own per-call cutoff.
describe('per-call options cannot be baked', () => {
  const bake = (options: unknown): unknown =>
    Reflect.apply(configure, undefined, [ratio, options])

  it('refuses scoreCutoff', () => {
    expect(() => bake({ scoreCutoff: 50 })).toThrow(TypeError)
  })

  it('refuses scoreHint', () => {
    expect(() => bake({ scoreHint: 50 })).toThrow(TypeError)
  })

  it('still accepts the options it can bake', () => {
    expect(() => bake({ processor: defaultProcess })).not.toThrow()
  })
})

describe('configuring a configured scorer', () => {
  it('merges the two option sets, outermost winning', () => {
    const once = configure(levenshteinDistance, { weights: [5, 5, 3] })
    const twice = configure(once, { weights: [5, 5, 1] })
    expect(twice('abc', 'abd')).toBe(1)
    expect(once('abc', 'abd')).toBe(3)
  })

  // Regression: the prepared factory a configured scorer registers used to bind
  // its own options and ignore the ones it was handed, so nesting silently used
  // the inner set through `process` while a direct call used the outer one.
  it('merges them on the prepared path too', () => {
    // Bound rather than nested inline: a generic call is not an inference
    // source for the generic call it is an argument to, so `configure` cannot
    // read the inner scorer's inputs out of an inline `configure(configure(…))`.
    const once = configure(levenshteinDistance, { weights: [5, 5, 3] })
    const twice = configure(once, { weights: [5, 5, 1] })
    const queries = ['abc', 'kitten']
    const choices = ['abd', 'sitting']
    expect(matrixScores(queries, choices, { scorer: twice })).toEqual(
      queries.map((q) => choices.map((c) => twice(q, c))),
    )
    expect(matrixScores(['abc'], ['abd'], { scorer: twice })[0][0]).toBe(1)
  })

  it('recomputes the flags from the merged options', () => {
    const symmetric = configure(levenshteinDistance, { weights: [1, 1, 2] })
    expect(scorerFlagsOf(symmetric).symmetric).toBe(true)
    expect(scorerFlagsOf(configure(symmetric, { weights: [1, 2, 1] })).symmetric).toBe(
      false,
    )
  })
})
