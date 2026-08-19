// Validating and compiling the `elementSimilarity` option: which inner scorers
// are admissible, and what a raw score becomes once rescaled onto `0..1`.
import { describe, expect, it } from 'vitest'

import { createScorer } from '#core/scoring/scorer.js'
import type { Direction } from '#core/types.js'

import { ratio as fuzzRatio } from '../../fuzz/index.js'
import { normalizedSimilarity as indelSimilarity } from '../indel/index.js'
import {
  compileElementSimilarity,
  CompiledElementSimilarity,
  effectiveElementSimilarity,
  elementScore,
  firstQualifyingFloat64,
} from './elementSimilarity.js'

const inner = createScorer(indelSimilarity)

function ordinalOf(value: number): bigint {
  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, value)
  const bits = view.getBigUint64(0)
  const sign = 0x8000_0000_0000_0000n
  return (bits & sign) === 0n ? bits ^ sign : bits ^ 0xffff_ffff_ffff_ffffn
}

function previousFloat(value: number): number {
  if (value === 0) return -Number.MIN_VALUE
  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, value)
  const bits = view.getBigUint64(0)
  view.setBigUint64(0, value > 0 ? bits - 1n : bits + 1n)
  return view.getFloat64(0)
}

function custom(
  bounds: readonly [number, number],
  extra: { direction?: Direction; symmetric?: boolean; missing?: 'throw' } = {},
) {
  const direction = extra.direction ?? 'similarity'
  const configuration = {
    direction,
    bounds,
    symmetric: extra.symmetric ?? true,
    ...(extra.missing === undefined ? {} : { missing: extra.missing }),
  }
  return createScorer((a, b) => (a === b ? bounds[1] : bounds[0]), configuration)
}

describe('compileElementSimilarity', () => {
  it('compiles a well-formed option', () => {
    const compiled = compileElementSimilarity({ scorer: inner, threshold: 0.8 }, 1)
    expect(compiled).toBeInstanceOf(CompiledElementSimilarity)
    expect(compiled.threshold).toBe(0.8)
    expect(compiled.lower).toBe(0)
    expect(compiled.span).toBe(1)
  })

  it('returns an already-compiled option unchanged', () => {
    const compiled = compileElementSimilarity({ scorer: inner, threshold: 0.8 }, 1)
    expect(compileElementSimilarity(compiled, 1)).toBe(compiled)
  })

  // The gram size is proved before idempotence, so a compiled option carried
  // into a gram-2 record is still refused. Nothing wants that pair to work, and
  // letting it through would turn a broken canonicalizer into a silent one.
  it('refuses an already-compiled option at the wrong gramSize', () => {
    const compiled = compileElementSimilarity({ scorer: inner, threshold: 0.8 }, 1)
    expect(() => compileElementSimilarity(compiled, 2)).toThrow(
      new RangeError('element similarity is only defined at gramSize 1'),
    )
  })

  it.each([2, 3])('refuses gramSize %i', (gramSize) => {
    expect(() =>
      compileElementSimilarity({ scorer: inner, threshold: 0.8 }, gramSize),
    ).toThrow(new RangeError('element similarity is only defined at gramSize 1'))
  })

  it.each([
    ['null', null],
    ['a number', 5],
    ['a string', 'jaro'],
    ['a function', () => 1],
  ])('refuses %s in place of an option object', (_label, value) => {
    expect(() => compileElementSimilarity(value, 1)).toThrow(TypeError)
  })

  it('refuses an unknown key', () => {
    expect(() =>
      compileElementSimilarity({ scorer: inner, threshold: 0.8, cutoff: 1 }, 1),
    ).toThrow(new TypeError("unknown elementSimilarity option 'cutoff'"))
  })

  it.each([
    ['missing', undefined],
    ['not scorer-shaped', { direction: 'similarity' }],
    ['a bare function', (a: string, b: string) => (a === b ? 1 : 0)],
  ])('refuses a scorer that is %s', (_label, scorer) => {
    expect(() => compileElementSimilarity({ scorer, threshold: 0.8 }, 1)).toThrow(
      new TypeError('elementSimilarity.scorer must be a scorer from createScorer'),
    )
  })

  it('refuses a scorer-shaped object createScorer did not make', () => {
    const impostor = {
      direction: 'similarity',
      bounds: [0, 1],
      symmetric: true,
      score: () => 1,
      prepareChoice: () => ({}),
    }
    expect(() =>
      compileElementSimilarity({ scorer: impostor, threshold: 0.8 }, 1),
    ).toThrow(new TypeError('scorer was not created by createScorer'))
  })

  it('refuses a distance scorer', () => {
    const distance = custom([0, 1], { direction: 'distance' })
    expect(() =>
      compileElementSimilarity({ scorer: distance, threshold: 0.8 }, 1),
    ).toThrow(new TypeError('elementSimilarity.scorer must be a similarity scorer'))
  })

  it('refuses an asymmetric scorer', () => {
    const asymmetric = custom([0, 1], { symmetric: false })
    expect(() =>
      compileElementSimilarity({ scorer: asymmetric, threshold: 0.8 }, 1),
    ).toThrow(new TypeError('elementSimilarity.scorer must be symmetric'))
  })

  // Core accepts all three; none of them can be rescaled. The last is the one
  // two finite endpoints do not rule out.
  it.each([
    ['zero-width', [1, 1] as const, { missing: 'throw' } as const],
    ['an infinite upper bound', [0, Number.POSITIVE_INFINITY] as const, {}],
    ['an infinite span', [-Number.MAX_VALUE, Number.MAX_VALUE] as const, {}],
  ])('refuses bounds with %s', (_label, bounds, extra) => {
    const scorer = custom(bounds, extra)
    expect(() => compileElementSimilarity({ scorer, threshold: 0.8 }, 1)).toThrow(
      new RangeError(
        'elementSimilarity.scorer needs finite bounds spanning a finite, non-zero range',
      ),
    )
  })

  it('accepts and rescales a scorer bounded 0..100', () => {
    const compiled = compileElementSimilarity(
      { scorer: createScorer(fuzzRatio), threshold: 0.8 },
      1,
    )
    expect(compiled.lower).toBe(0)
    expect(compiled.span).toBe(100)
  })

  it.each([
    ['missing', undefined],
    ['null', null],
    ['a string', '0.8'],
  ])('refuses a threshold that is %s', (_label, threshold) => {
    expect(() => compileElementSimilarity({ scorer: inner, threshold }, 1)).toThrow(
      new TypeError('elementSimilarity.threshold must be a number'),
    )
  })

  it.each([
    ['NaN', Number.NaN],
    ['infinite', Number.POSITIVE_INFINITY],
    ['zero', 0],
    ['negative', -0.5],
    ['above one', 1.000_000_000_000_000_2],
  ])('refuses a threshold that is %s', (_label, threshold) => {
    expect(() => compileElementSimilarity({ scorer: inner, threshold }, 1)).toThrow(
      new RangeError('elementSimilarity.threshold has to be above 0 and at most 1'),
    )
  })

  it('accepts a threshold of exactly 1', () => {
    expect(compileElementSimilarity({ scorer: inner, threshold: 1 }, 1).threshold).toBe(1)
  })
})

describe('native element threshold', () => {
  it.each([
    [0, 1, Number.MIN_VALUE],
    [0, 1, 0],
    [0, 1, 1],
    [-1, 1, 0.5],
    [-1000, -999, 0.9],
    [-Number.MIN_VALUE, Number.MIN_VALUE, 0.5],
    [-1e16, 1, Number.MIN_VALUE],
    [-1e16, 1, 0.5],
    [-1e300, 1e300, 0.5],
    [1e300, 1e300 + 1e284, 0.25],
  ])('finds the first qualifying Float64 in [%d, %d]', (lower, upper, threshold) => {
    const span = upper - lower
    const raw = firstQualifyingFloat64(lower, upper, threshold, span)
    expect((raw - lower) / span).toBeGreaterThanOrEqual(threshold)
    if (raw !== lower) {
      expect((previousFloat(raw) - lower) / span).toBeLessThan(threshold)
    }
  })

  it('lands astronomically far from the lower bound, where walking cannot reach', () => {
    // The reason this is a binary search over ordered bits and not a `nextUp`
    // walk: the answer sits over 10^17 representable values above `lower`, so
    // an adjacent-value walk would never terminate. Replacing the search with
    // one must fail this test rather than merely run slowly.
    const lower = -1e300
    const upper = 1e300
    const raw = firstQualifyingFloat64(lower, upper, 0.5, upper - lower)
    expect(ordinalOf(raw) - ordinalOf(lower)).toBeGreaterThan(2n ** 50n)
    expect((raw - lower) / (upper - lower)).toBeGreaterThanOrEqual(0.5)
    expect((previousFloat(raw) - lower) / (upper - lower)).toBeLessThan(0.5)
  })
})

describe('effectiveElementSimilarity', () => {
  it('is null where the caller asked for exact matching', () => {
    expect(effectiveElementSimilarity(undefined, 2)).toBeNull()
  })

  it('compiles anything else', () => {
    const compiled = effectiveElementSimilarity({ scorer: inner, threshold: 0.5 }, 1)
    expect(compiled).toBeInstanceOf(CompiledElementSimilarity)
  })
})

describe('elementScore', () => {
  it('reports an exact pair as 1 and rescales a 0..100 scorer', () => {
    const rescaled = compileElementSimilarity(
      { scorer: createScorer(fuzzRatio), threshold: 0.5 },
      1,
    )
    expect(elementScore(rescaled, 'swisscom', 'swisscom')).toBe(1)
    const near = elementScore(rescaled, 'swisscom', 'swisscomm')
    expect(near).toBeGreaterThan(0.5)
    expect(near).toBeLessThan(1)
  })

  it('stays within 0..1 for a scorer whose lower bound is not 0', () => {
    const shifted = custom([-1, 1])
    const compiled = compileElementSimilarity({ scorer: shifted, threshold: 0.5 }, 1)
    expect(elementScore(compiled, 'ab', 'ab')).toBe(1)
    expect(elementScore(compiled, 'ab', 'cd')).toBe(0)
  })
})
